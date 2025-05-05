"""Get messages from desired LCM channels from an LCM log between provided
start and end times.  Uses the virtual environment installed at
ci_mpc_utils/venv.

source /home/bibit/ci_mpc_utils/venv/bin/activate

Example usage:

# TODO:  `single` command is currently broken.
python lcm_log_processing.py single /mnt/data2/sharanya/logs/2024/12_16_24/000006/ --interactive --start=50 --end=100
python lcm_log_processing.py multi /home/bibit/Videos/franka_experiments/2025/01_29_25/000007
"""

import click
import os
import os.path as op
import pickle
import shutil
import subprocess
from tqdm import tqdm
from typing import List, Tuple

from lcm import EventLog
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import gaussian_kde
import yaml

from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import HalfSpace, MeshcatVisualizer, StartMeshcat, \
    ClippingRange, DepthRange, DepthRenderCamera, \
    RenderCameraCore, MakeRenderEngineVtk, RenderEngineVtkParams
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant, MultibodyPlantConfig
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import CameraInfo, RgbdSensor
from pydrake.trajectories import PiecewisePolynomial, \
    PiecewiseQuaternionSlerp, StackedTrajectory
from pydrake.visualization import VideoWriter

import file_utils
file_utils.add_dair_lcmtypes_to_path()

import dairlib


# Add to this dictionary to include more LCM channels from which to read.
ALL_CHANNELS_AND_LCMT = {
    'C3_ACTUAL': dairlib.lcmt_c3_state,
    'C3_FINAL_TARGET': dairlib.lcmt_c3_state,
    'SAMPLE_BUFFER': dairlib.lcmt_sample_buffer,
    'SAMPLE_LOCATIONS': dairlib.lcmt_timestamped_saved_traj,
    'SAMPLE_COSTS': dairlib.lcmt_timestamped_saved_traj,
    'C3_TRAJECTORY_ACTOR_CURR_PLAN': dairlib.lcmt_timestamped_saved_traj,
    'C3_TRAJECTORY_ACTOR_BEST_PLAN': dairlib.lcmt_timestamped_saved_traj,
    'SAMPLING_C3_DEBUG': dairlib.lcmt_sampling_c3_debug,
    'SAMPLING_C3_RADIO': dairlib.lcmt_radio_out,
    'FRANKA_STATE': dairlib.lcmt_robot_output,
    'FRANKA_STATE_SIMULATION': dairlib.lcmt_robot_output,
    'OBJECT_STATE': dairlib.lcmt_object_state,
}
MINIMAL_CHANNELS_AND_LCMT = {
    'SAMPLE_BUFFER': dairlib.lcmt_sample_buffer,
    'SAMPLING_C3_DEBUG': dairlib.lcmt_sampling_c3_debug,
}
MINIMAL_CHANNELS_AND_LCMT_FOR_VIDEO = {
    'SAMPLE_BUFFER': dairlib.lcmt_sample_buffer,
    'SAMPLING_C3_DEBUG': dairlib.lcmt_sampling_c3_debug,
    'C3_ACTUAL': dairlib.lcmt_c3_state,
    'C3_FINAL_TARGET': dairlib.lcmt_c3_state,
}
MINIMAL_CHANNELS_AND_LCMT_FOR_MJPC = {
    'C3_ACTUAL': dairlib.lcmt_c3_state,
    'C3_FINAL_TARGET': dairlib.lcmt_c3_state,
}
CHANNELS_AND_LCMT_TO_SYNC = {
    key: val for key, val in ALL_CHANNELS_AND_LCMT.items() \
    if key not in ['SAMPLING_C3_RADIO', 'FRANKA_STATE', 'OBJECT_STATE']}
LCM_TIME_KEY = 'lcm_seconds'
MESSAGE_TIME_KEY = 'msg_seconds'
MESSAGE_KEY = 'message'

JOINT_LIMIT_VIO_KEY = 'joint_limit_violation'
JOINT_VEL_VIO_KEY = 'joint_velocity_violation'
JOINT_ACC_VIO_KEY = 'joint_acceleration_violation'
JOINT_JERK_VIO_KEY = 'joint_jerk_violation'
JOINT_TORQUE_VIO_KEY = 'joint_torque_violation'
WSL_VIO_KEY = 'workspace_limit_violation'

TRAJ_PARAM_POS_TOL_KEY = 'position_success_threshold'
TRAJ_PARAM_RAD_TOL_KEY = 'orientation_success_threshold'

C3_PARAM_WSL_X_KEY = 'world_x_limits'
C3_PARAM_WSL_Y_KEY = 'world_y_limits'
C3_PARAM_WSL_Z_KEY = 'world_z_limits'
C3_PARAM_WSL_R_KEY = 'robot_radius_limits'

EXPORT_FOLDER = '/mnt/data2/bibit/control_exports'

# Labels for the source of repositioning targets.
NO_TARGET_LABEL = 'N/A'
PREV_REPOS_TARGET_LABEL = 'previous repositioning target'
NEW_SAMPLE_TARGET_LABEL = 'new sample target'
BUFFER_SAMPLE_TARGET_LABEL = 'buffer sample target'
PURSUED_TARGET_LABELS = [NO_TARGET_LABEL,
                         PREV_REPOS_TARGET_LABEL,
                         NEW_SAMPLE_TARGET_LABEL,
                         BUFFER_SAMPLE_TARGET_LABEL]

# Labels for the reason behind switching modes.
NO_SWITCH_LABEL = 'N/A'
TO_C3_LOWER_COST_SWITCH_LABEL = 'Switch to Contact-Rich:  lower cost'
TO_C3_REACHED_TARGET_SWITCH_LABEL = \
    'Switch to Contact-Rich:  reached pursued sample'
TO_REPOS_LOWER_COST_SWITCH_LABEL = 'Switch to Contact-Free:  lower cost'
TO_REPOS_UNPRODUCTIVE_SWITCH_LABEL = 'Switch to Contact-Free:  unproductivity'
TO_C3_XBOX_FORCED_SWITCH_LABEL = 'Switch to Contact-Rich:  Xbox'
MODE_SWITCH_LABELS = [NO_SWITCH_LABEL,
                      TO_C3_LOWER_COST_SWITCH_LABEL,
                      TO_C3_REACHED_TARGET_SWITCH_LABEL,
                      TO_REPOS_LOWER_COST_SWITCH_LABEL,
                      TO_REPOS_UNPRODUCTIVE_SWITCH_LABEL,
                      TO_C3_XBOX_FORCED_SWITCH_LABEL]
MODE_SWITCH_COLORS = ['black', '#f78b8e', '#ffcc80', '#a5c493', '#c495f0',
                      'purple']
CF_COLOR = '#bbbbbb'
CR_COLOR = 'white'
POS_ERROR_COLOR = 'black'
RAD_ERROR_COLOR = 'purple'

# Other success thresholds.
# POS_SUCCESS_THRESHOLDS = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
# RAD_SUCCESS_THRESHOLDS = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
# THRESHOLD_COLORS = ['black', 'red', 'darkorange', 'gold', 'green']
POS_SUCCESS_THRESHOLDS = np.array([0.02, 0.05])
RAD_SUCCESS_THRESHOLDS = np.array([0.1, 0.4])
THRESHOLD_COLORS = ['red', 'blue']

EPS = 1e-5
TIME_SYNCH_THRESH = 0.03
HIST_BINS = 10
CDF_TIME_CUTOFF = 300
TOO_LONG_TIME_CUTOFF = 400

CAM_FOV = np.pi/6
VIDEO_PIXELS = [480, 640]
VIDEO_FPS = 30

# Front video view (for the jack).
SENSOR_RPY_FRONT = np.array([-np.pi / 2, 0, np.pi / 2])
SENSOR_POSITION_FRONT = np.array([2., 0., 0.2])
SENSOR_POSE_FRONT_VIEW = RigidTransform(
    RollPitchYaw(SENSOR_RPY_FRONT).ToQuaternion(), SENSOR_POSITION_FRONT)
JACK_LOCKED_CAMERA_OFFSET = np.array([0.6, 0, 0]).reshape(3, 1)
JACK_LOCKED_CAMERA_QUAT = SENSOR_POSE_FRONT_VIEW.rotation().ToQuaternion(
    ).wxyz().reshape(4, 1)

# Top video view (for the T).
SENSOR_RPY_TOP = np.array([np.pi, 0, np.pi / 2])
SENSOR_POSITION_TOP = np.array([0., 0., 2.])
SENSOR_POSE_TOP_VIEW = RigidTransform(
    RollPitchYaw(SENSOR_RPY_TOP).ToQuaternion(), SENSOR_POSITION_TOP)
T_LOCKED_CAMERA_OFFSET = np.array([0, 0, 0.8]).reshape(3, 1)
T_LOCKED_CAMERA_QUAT = SENSOR_POSE_TOP_VIEW.rotation().ToQuaternion(
    ).wxyz().reshape(4, 1)

ACTUAL_JACK_URDF_PATH = file_utils.jack_with_triad_urdf_path()
GOAL_JACK_URDF_PATH = file_utils.goal_triad_urdf_path()
ACTUAL_T_URDF_PATH = file_utils.push_t_urdf_path()
GOAL_T_URDF_PATH = file_utils.goal_push_t_urdf_path()
CAMERA_URDF_PATH = file_utils.camera_urdf_path()
SECOND_CAMERA_URDF_PATH = file_utils.camera_urdf_path(first=False)
# ACTUAL_JACK_URDF_PATH = 'examples/jacktoy/urdf/jack_with_triad.urdf'
# GOAL_JACK_URDF_PATH = 'examples/jacktoy/urdf/goal_triad.urdf'
# ACTUAL_T_URDF_PATH = 'examples/push_T/urdf/T_vertical_obj.urdf'
# GOAL_T_URDF_PATH = 'examples/push_T/urdf/T_vertical_obj_green.urdf'
# CAMERA_URDF_PATH = 'examples/jacktoy/urdf/camera_model.urdf'
# SECOND_CAMERA_URDF_PATH = 'examples/jacktoy/urdf/camera_model_2.urdf'

BOTTOM_LEFT_PLACEMENT = '1400:690'
BOTTOM_RIGHT_PLACEMENT = '20:690'
BOTTOM_LEFT_BOX_PLACEMENT = 'x=10:y=680'
BOTTOM_RIGHT_BOX_PLACEMENT = 'x=1380:y=680'
UPPER_LEFT_TEXT_PLACEMENT = 'x=50:y=50'
UPPER_RIGHT_TEXT_PLACEMENT = 'x=1200:y=50'

# Franka limits, from:
# https://frankaemika.github.io/docs/control_parameters.html#limits-for-panda
FRANKA_JOINT_MINS = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8976, -0.0175, -2.8973])
FRANKA_JOINT_MAXS = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8976, 3.7525, 2.8973])
FRANKA_JOINT_VEL_LIMITS = np.array(
    [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
FRANKA_JOINT_ACC_LIMITS = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
FRANKA_JOINT_JERK_LIMITS = np.array(
    [7500, 3750, 5000, 6250, 7500, 10000, 10000])
FRANKA_JOINT_TORQUE_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12])
# An alternative source, from:
# https://frankaemika.github.io/docs/franka_ros.html#franka-control
# FRANKA_JOINT_TORQUE_LIMITS = np.array([20, 20, 18, 18, 16, 14, 12])

# Keyed by the log file, has a tuple of the long video filepath and a directory
# to which single-goal videos can be written.
LOG_FILEPATHS_TO_VIDEOS = {
    # Jack hardware videos, random goals with tight tolerances:
    '/mnt/data2/sharanya/logs/2025/01_29_25/000007/hwlog-000007':
        (op.join('/mnt/data2/sharanya/Hardware_videos_Sharanya/Jan29/Dump1',
                 '20250129_124229_log7_cut_2.mp4'),
         '/mnt/data2/bibit/log_videos/01_29_25_log_7/'),
    '/mnt/data2/sharanya/logs/2025/01_29_25/000031/hwlog-000031':
        (op.join('/mnt/data2/sharanya/Hardware_videos_Sharanya/Jan29/Dump2',
                '20250129_162050_log31_cut_2.mp4'),
         '/mnt/data2/bibit/log_videos/01_29_25_log_31/'),
    '/mnt/data2/sharanya/logs/2025/01_29_25/000032/hwlog-000032':
        (op.join('/mnt/data2/sharanya/Hardware_videos_Sharanya/Jan29/Dump2',
                '20250129_170022_log32_cut.mp4'),
         '/mnt/data2/bibit/log_videos/01_29_25_log_32/'),
    '/mnt/data2/sharanya/logs/2025/01_29_25/000033/hwlog-000033':
        (op.join('/mnt/data2/sharanya/Hardware_videos_Sharanya/Jan29/Dump2',
                '20250129_173127_log33_cut_2.mp4'),
         '/mnt/data2/bibit/log_videos/01_29_25_log_33/'),
    # Jack hardware videos, orientation cycling with loose tolerances:
    '/mnt/data2/sharanya/logs/2025/01_14_25/000029/hwlog-000029':
        (op.join('/mnt/data2/sharanya/Hardware_videos_Sharanya/',
                '20250114_log29_cut_2.MOV'),
         '/mnt/data2/bibit/log_videos/01_14_25_log_29/'),
    # # Box sim videos:
    # '/mnt/data2/sharanya/logs/2025/01_24_25/000021/hwlog-000021':
    #     ('/mnt/data2/sharanya/sim_videos/Jan24_log21_box.webm',
    #      '/mnt/data2/bibit/log_videos/01_24_25_log_21/'),
    # '/mnt/data2/sharanya/logs/2025/01_24_25/000032/hwlog-000032':
    #     ('/mnt/data2/sharanya/sim_videos/Jan24_log32_box.webm',
    #      '/mnt/data2/bibit/log_videos/01_24_25_log_32/'),
    # One for a local test:
    '/home/bibit/Videos/franka_experiments/2025/01_29_25/000007/hwlog-000007':
        (op.join('/home/bibit/Videos/franka_experiments/2025/01_29_25',
                 '20250129_124229_log7_cut.mp4'),
         '/home/bibit/Videos/franka_experiments/2025/01_29_25/'),
    # Push T hardware videos, all tight tolerances:
    '/mnt/data2/sharanya/logs/2025/03_19_25/000016/hwlog-000016':
        (None,  # TODO this recording must be on Sharanya's phone
         '/mnt/data2/bibit/log_videos/03_19_25_log_16/'),
    '/mnt/data2/sharanya/logs/2025/03_19_25/000018/hwlog-000018':
        ('/mnt/data2/bibit/hardware_videos/03_19_2025_log_18_trimmed.MOV',
         '/mnt/data2/bibit/log_videos/03_19_25_log_18/'),
    '/mnt/data2/sharanya/logs/2025/03_19_25/000024/hwlog-000024':
        ('/mnt/data2/bibit/hardware_videos/03_19_2025_log_24_trimmed.MOV',
         '/mnt/data2/bibit/log_videos/03_19_25_log_24/'),
    '/mnt/data2/sharanya/logs/2025/03_19_25/000029/hwlog-000029':
        ('/mnt/data2/bibit/hardware_videos/03_19_2025_log_29_trimmed.MOV',
         '/mnt/data2/bibit/log_videos/03_19_25_log_29/'),
}

PUSH_T_LOG_PATHS = [
    '/mnt/data2/sharanya/logs/2025/03_19_25/000016/hwlog-000016',
    '/mnt/data2/sharanya/logs/2025/03_19_25/000018/hwlog-000018',
    '/mnt/data2/sharanya/logs/2025/03_19_25/000024/hwlog-000024',
    '/mnt/data2/sharanya/logs/2025/03_19_25/000029/hwlog-000029'
]
JACK_LOG_PATHS = list(LOG_FILEPATHS_TO_VIDEOS.keys())
for push_t_path in PUSH_T_LOG_PATHS:
    if push_t_path in JACK_LOG_PATHS:  JACK_LOG_PATHS.remove(push_t_path)

MJPC_CUT_OFF_GOALS_PER_EE_VEL = {
    0.03: 0, 0.06: 0, 0.09: 0, 0.12: 2, 0.15: 0, 0.18: 1, 0.21: 1, 0.24: 7,
    0.27: 5, 1000: 0
}
MJPC_COLORS_PER_EE_VEL = {
    0.03: '#fedbda',
    0.06: '#fdb7b5',
    0.09: '#fc9490',
    0.12: '#fb706a',
    0.15: '#fb4d46',
    0.18: '#c83d38',
    0.21: '#962e2a',
    0.24: '#641e1c',
    0.27: '#320f0e',
    1000: '#30ba8f',  # For our controller.
}

global_is_interactive = False


def get_date_and_log_num_from_log_filepath(log_filepath: str) -> Tuple[str,
                                                                       int]:
    """Returns the date and log number as strings."""
    log_folder, log_filename = op.split(log_filepath)
    log_num = int(log_filename.split('-')[-1])
    date_str = op.split(op.split(log_folder)[0])[-1]
    return date_str, log_num

def get_video_duration(filepath: str) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def save_current_figure(filename: str, store_folder: str = None):
    if store_folder is not None:
        filepath = op.join(store_folder, f'{filename}.png')
    else:
        filepath = op.join(file_utils.tmp_dir(), f'{filename}.png')
    plt.savefig(filepath)
    print(f'Wrote plot to {filepath}')

    global global_is_interactive
    if not global_is_interactive:
        plt.close()

def get_shading_masks(bool_array):
    bool_array = bool_array.squeeze()
    assert bool_array.ndim == 1
    bool_array = bool_array.astype(bool)
    right_shifted_yes = np.append(bool_array[0], bool_array[:-1])

    yes_shading_mask = np.ravel(np.column_stack(
        (right_shifted_yes, bool_array)))
    no_shading_mask = np.ravel(np.column_stack(
        (~right_shifted_yes, ~bool_array)))

    return yes_shading_mask, no_shading_mask

# TODO implement and remove
def visualize_sample_buffer(messages_by_channel: dict, log_folder: str = None):
    # First start with just a simple matplotlib plot of the buffer contents.
    sample_buffers = messages_by_channel['SAMPLE_BUFFER'][MESSAGE_KEY]
    states = messages_by_channel['C3_ACTUAL'][MESSAGE_KEY]
    debugs = messages_by_channel['SAMPLING_C3_DEBUG'][MESSAGE_KEY]

    times = messages_by_channel['SAMPLE_BUFFER'][LCM_TIME_KEY]

    n_in_buffers = []
    quats, xyzs, ee_xyzs = [], [], []
    # Store orientation error in full rotation units for easier viewing on same
    # axis as meter distance error.
    pos_errors, full_rotation_errors = [], []
    for buffer, state, debug in zip(sample_buffers, states, debugs):
        n_in_buffers.append(buffer.num_in_buffer)
        quats.append(state.state[3:7])
        xyzs.append(state.state[7:10])
        ee_xyzs.append(state.state[:3])

        pos_errors.append(debug.current_pos_error)
        full_rotation_errors.append(debug.current_rot_error / (2*np.pi))

    _fig, axs = plt.subplots(2, 1, figsize=(6, 9), sharex=True)
    axs[0].plot(times, n_in_buffers)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Number of samples')
    axs[0].set_title('Number of active samples in buffer')
    axs[1].plot(times, pos_errors, label='Position error')
    axs[1].plot(times, full_rotation_errors, label='Rotation error')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Error [m or full rotation]')
    axs[1].set_title('Error between current and goal states')
    plt.legend()
    save_current_figure('sample_buffer', store_folder=log_folder)

def inspect_mode_switching_by_goal(times: np.ndarray,
                                   is_c3_mode_flags: np.ndarray,
                                   switch_reasons: list,
                                   pos_errors: np.ndarray,
                                   rad_errors: np.ndarray,
                                   pos_tol: float, rad_tol: float,
                                   goal_num: int,
                                   log_folder: str = None):
    print(f'Making plot for goal {goal_num}...')

    times = times - times[0]
    double_t = np.repeat(times, 2)
    if is_c3_mode_flags is not None:
        c3_mask, repos_mask = get_shading_masks(is_c3_mode_flags)
    deg_errors = rad_errors * 180 / np.pi

    # A more presentable plot:  position and rotation errors with mode shading
    # and mode switch lines.
    fig, axs = plt.subplots(1, 1, figsize=(14.3, 4))
    ax0 = axs
    ax1 = ax0.twinx()
    if is_c3_mode_flags is not None:
        ax0.fill_between(double_t, 0, 1.05, where=repos_mask, color=CF_COLOR,
                        alpha=0.5, transform=ax0.get_xaxis_transform())
        ax0.fill_between(double_t, 0, 1.05, where=c3_mask, color=CR_COLOR,
                        alpha=0.5, transform=ax0.get_xaxis_transform())

    # Add mode switch lines.
    if switch_reasons is not None:
        for i, mode_switch_reason in enumerate(MODE_SWITCH_LABELS):
            if i == 0:
                continue
            switch_ts = np.array(times)[np.array(switch_reasons) == i]
            prefix = ''
            for switch_t in switch_ts:
                ax0.axvline(x=switch_t, color=MODE_SWITCH_COLORS[i],
                            linewidth=4, label=prefix + mode_switch_reason)
                prefix = '_'

    ax0.plot(times, pos_errors, color=POS_ERROR_COLOR, label='Position error')
    ax1.plot(times, deg_errors, color=RAD_ERROR_COLOR,
             label='Orientation error')

    ax0.axhline(y=pos_tol, linestyle='--', color=POS_ERROR_COLOR,
                   label='Position success threshold')
    ax1.axhline(y=rad_tol * 180/np.pi, linestyle='--', color=RAD_ERROR_COLOR,
                label='Orientation success threshold')

    ax0.set_xlabel('Time (s)', fontsize=16)
    ax0.set_ylabel('Position Error [m]', fontsize=16)
    ax0.set_ylim([0, np.max(pos_errors) + 0.01])
    ax0.set_xlim([np.min(times), np.max(times)])
    ax1.set_ylabel('Orientation Error [deg]', color=RAD_ERROR_COLOR, fontsize=16)
    ax1.set_ylim([0, np.max(deg_errors) + 10])
    ax1.tick_params(axis='y', labelcolor=RAD_ERROR_COLOR)
    fig.suptitle(f'Errors over Time with Mode Switching for Goal {goal_num}', fontsize=18)

    # Legend:  need to add patches manually.
    cf_patch = Patch(facecolor=CF_COLOR, alpha=0.5, edgecolor='black',
                     linewidth=1, label='Contact-free mode')
    cr_patch = Patch(facecolor=CR_COLOR, alpha=0.5, edgecolor='black',
                     linewidth=1, label='Contact-rich mode')

    ax0.legend(
        handles=ax0.get_legend_handles_labels()[0] + \
            ax1.get_legend_handles_labels()[0] + [cf_patch, cr_patch],
        bbox_to_anchor=(1.2, 1), loc='upper left', fontsize=14,
        title_fontsize=16)
    plt.tight_layout()
    save_current_figure(
        f'shading_goal_{goal_num}' if log_folder is not None else \
        'shading_goal', store_folder=log_folder)

# TODO implement and remove
def inspect_lcm_traffic(messages_by_channel: dict):
    buffer_ts = messages_by_channel['SAMPLE_BUFFER'][LCM_TIME_KEY]
    franka_ts = messages_by_channel['FRANKA_STATE'][LCM_TIME_KEY]
    radio_ts = messages_by_channel['SAMPLING_C3_RADIO'][LCM_TIME_KEY]

    buffer_ts = np.array(buffer_ts)
    franka_ts = np.array(franka_ts)
    radio_ts = np.array(radio_ts)

    print(f'Max buffer time: {np.max(buffer_ts)}')
    print(f'Max Franka time: {np.max(franka_ts)}')
    print(f'Max radio time: {np.max(radio_ts)}')

    plt.figure()
    plt.plot(buffer_ts, label='Buffer times')
    plt.plot(franka_ts, label='Franka times')
    plt.plot(radio_ts, label='Radio times')
    plt.legend()
    plt.show()

def log_is_push_t(log_path: str):
    if log_path in JACK_LOG_PATHS:
        return False
    if log_path in PUSH_T_LOG_PATHS:
        return True
    raise ValueError(f'Unsure if {log_path} is push T or jack log.')

def relative_angle_from_quats(q, r):
    q /= np.linalg.norm(q)
    r /= np.linalg.norm(r)
    qr = R.from_quat([q[1], q[2], q[3], q[0]])
    rr = R.from_quat([r[1], r[2], r[3], r[0]])
    angle = (qr * rr.inv()).magnitude()
    return angle

def joint_mjpc_cdf(ras_by_ee_vel: dict, our_ra):
    ee_vels = np.array(list(ras_by_ee_vel.keys()) + [1000])
    ras = np.array(list(ras_by_ee_vel.values()) + [our_ra])

    # Sort based on shortest to longest time to goal.
    sorted_idx = ee_vels.argsort()
    sorted_ee_vels = ee_vels[sorted_idx]
    sorted_ras = ras[sorted_idx]

    # Plot formatting.
    plt.rcParams.update({'font.family': 'serif'})

    # Generate a plot of cumulative distribution functions.
    fig, axs = plt.subplots(1, 1, figsize=(10,4))
    for ee_vel, ra in zip(sorted_ee_vels, sorted_ras):
        color = MJPC_COLORS_PER_EE_VEL[ee_vel]

        # Get the tightest threshold data.
        data = ra.times_to_thresholds[:, 0]

        # Compute the fraction of invalid goals.
        problem_goals = []
        problem_goals += ra.goal_violations[JOINT_LIMIT_VIO_KEY]
        problem_goals += ra.goal_violations[JOINT_VEL_VIO_KEY]
        problem_goals += ra.goal_violations[JOINT_TORQUE_VIO_KEY]
        valid_goal_mask = np.array([i not in problem_goals for i in range(
            1, ra.n_goals_achieved[-1].item() + 2)])
        invalid_fraction = (~valid_goal_mask).sum()/valid_goal_mask.shape[0]
        trials = len(data)
        cut_off = MJPC_CUT_OFF_GOALS_PER_EE_VEL[ee_vel]
        hw_viols = f'{100*invalid_fraction:.1f}%'

        # Include the few goals that were manually cut off after > 400s.
        for _ in range(MJPC_CUT_OFF_GOALS_PER_EE_VEL[ee_vel]):
            data = np.concatenate((data, np.array([400])))

        count, bins_count = np.histogram(
            data, bins=11, range=(0, TOO_LONG_TIME_CUTOFF))
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        bins_count[0] = 0
        cdf = np.concatenate([[0], cdf])
        label = f'MJPC {ee_vel:.2f}' if ee_vel < 1000 else 'Ours         '
        label += f' / {trials} / {cut_off} / {hw_viols}'
        axs.plot(bins_count, cdf, color=color, linewidth=5, label=label)

    axs.set_xlabel('Time Limit [s]', fontsize=16)
    axs.set_ylabel('Fraction of Trials', fontsize=16)
    axs.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axs.set_yticks(np.linspace(0, 1, 11))
    axs.set_ylim([0, 1])
    axs.set_xlim([0, CDF_TIME_CUTOFF])
    axs.tick_params(axis='both', which='major', labelsize=12)
    fig.suptitle('Fraction of Goals Achieved Within Time Limit', fontsize=18)

    axs.legend(bbox_to_anchor=(1.02, 1.04), loc='upper left',
               title='Approach / AG\u2191 / COG\u2193 / HWV\u2193', fontsize=14,
               title_fontsize=15)
    plt.tight_layout()

    plt.grid()
    save_current_figure('mjpc_cdf', store_folder=ra.save_folder)


class ResultsAnalyzer:
    """Analyzes the results of a sampling-based C3 or MuJoCo MPC log by
    extracting information out of one or multiple associated LCM logs, with the
    ability to generate visuals."""
    def __init__(self, log_filepaths: List[str], channels: List[str],
                 sync_channels: List[str] = None,
                 start_times: List[float] = None, end_times: List[float] = None,
                 save_folder: str = None, verbose: bool = True,
                 trim_bookends: bool = False):
        assert len(channels) > 0, 'Need at least one channel to visualize.'
        if start_times is not None:
            assert len(start_times) == len(log_filepaths), f'Need either no' + \
                f' or all start times.'
        else:
            start_times = [None] * len(log_filepaths)
        if end_times is not None:
            assert len(end_times) == len(log_filepaths), f'Need either no ' + \
                f'or all end times.'
        else:
            end_times = [None] * len(log_filepaths)

        self.log_filepaths = log_filepaths
        self.start_times = start_times
        self.end_times = end_times
        self._channels = channels if type(channels) == list else list(channels)
        self.save_folder = save_folder
        self.trim_bookends = trim_bookends

        self._use_debug_instead_of_target = 'SAMPLING_C3_DEBUG' in channels

        # Load and stitch together each log file.
        self.lcm_t_adj = 0
        self.msg_t_adj = 0
        self.messages_by_channel = {
            key: {LCM_TIME_KEY: [], MESSAGE_TIME_KEY: [], MESSAGE_KEY: []}
            for key in self._channels
        }

        # The start LCM times per log will be one longer than the number of logs
        # since it includes the end time of the last log.
        self.log_lcm_start_times = [self.lcm_t_adj]
        self.goals_per_log = []
        for log_file, start, end in zip(log_filepaths, start_times, end_times):
            self._add_messages_from_log(
                log_file, start_time=start, end_time=end, verbose=verbose)
            self.log_lcm_start_times.append(self.lcm_t_adj)

        # Synchronize according to channels.
        self._synchronize_messages(
            sync_channels=sync_channels, visualize=verbose)

    def _get_trajectory_tolerances(self):
        """Get the position and orientation tolerances from the trajectory
        parameters files, ensuring that all logs have the same tolerances.
        Store them as self.pos_tol and self.rad_tol."""
        if hasattr(self, 'pos_tol') and hasattr(self, 'rad_tol'):
            return

        pos_tol = None
        rad_tol = None

        # Get the trajectory parameters.
        for log_filepath in self.log_filepaths:
            log_folder, log_filename = op.split(log_filepath)
            log_number = log_filename.split('-')[-1]

            trajectory_params_filepath = op.join(
                log_folder, f'trajectory_params_{log_number}.yaml')
            with open(trajectory_params_filepath, 'r') as file:
                traj_params = yaml.safe_load(file)

            if pos_tol is None:
                pos_tol = traj_params[TRAJ_PARAM_POS_TOL_KEY]
                rad_tol = traj_params[TRAJ_PARAM_RAD_TOL_KEY]

            else:
                assert pos_tol == traj_params[TRAJ_PARAM_POS_TOL_KEY], \
                    'Position success thresholds do not match: ' + \
                    f'{pos_tol} vs. {traj_params[TRAJ_PARAM_POS_TOL_KEY]}'
                assert rad_tol == traj_params[TRAJ_PARAM_RAD_TOL_KEY], \
                    'Orientation success thresholds do not match: ' + \
                    f'{rad_tol} vs. {traj_params[TRAJ_PARAM_RAD_TOL_KEY]}'

        self.pos_tol = pos_tol
        self.rad_tol = rad_tol

    def _get_workspace_limits(self):
        """Get the x, y, z, and radius limits from the C3 parameter files,
        ensuring that all logs have the same workspace limits.  Store them as
        self.wsl_x, wsl_y, wsl_z, and wsl_r."""
        if hasattr(self, 'wsl_x'):
            return

        wsl_x = None
        wsl_y = None
        wsl_z = None
        wsl_r = None

        # Get the C3 options.
        for log_filepath in self.log_filepaths:
            log_folder, log_filename = op.split(log_filepath)
            log_number = log_filename.split('-')[-1]

            c3_params_filepath = op.join(
                log_folder, f'c3_gains_{log_number}.yaml')
            with open(c3_params_filepath, 'r') as file:
                c3_params = yaml.safe_load(file)

            if wsl_x is None:
                wsl_x = c3_params[C3_PARAM_WSL_X_KEY]
                wsl_y = c3_params[C3_PARAM_WSL_Y_KEY]
                wsl_z = c3_params[C3_PARAM_WSL_Z_KEY]
                wsl_r = c3_params[C3_PARAM_WSL_R_KEY]

            else:
                assert wsl_x == c3_params[C3_PARAM_WSL_X_KEY], \
                    'X workspace limits do not match: ' + \
                    f'{wsl_x} vs. {c3_params[C3_PARAM_WSL_X_KEY]}'
                assert wsl_y == c3_params[C3_PARAM_WSL_Y_KEY], \
                    'X workspace limits do not match: ' + \
                    f'{wsl_y} vs. {c3_params[C3_PARAM_WSL_Y_KEY]}'
                assert wsl_z == c3_params[C3_PARAM_WSL_Z_KEY], \
                    'X workspace limits do not match: ' + \
                    f'{wsl_z} vs. {c3_params[C3_PARAM_WSL_Z_KEY]}'
                assert wsl_r == c3_params[C3_PARAM_WSL_R_KEY], \
                    'X workspace limits do not match: ' + \
                    f'{wsl_r} vs. {c3_params[C3_PARAM_WSL_R_KEY]}'

        self.wsl_x = wsl_x
        self.wsl_y = wsl_y
        self.wsl_z = wsl_z
        self.wsl_r = wsl_r

    def _add_messages_from_log(self, log_filepath: str, start_time: float = 0.0,
                               end_time: float = 1e12, verbose: bool = True):
        """Add messages and times for every channel of interest into the
        current self.messages_and_channels dictionary, appending the new data
        to the end as if it were a continuous experiment."""
        start_time = 0.0 if start_time is None else start_time
        end_time = 1e12 if end_time is None else end_time

        start_utime = int(start_time*1e6)
        end_utime = int(end_time*1e6)

        # Open the LCM log.
        log_file = EventLog(log_filepath, 'r')

        # Read through the log file.
        event = log_file.read_next_event()
        while event.channel not in self._channels:
            event = log_file.read_next_event()
        init_lcm_utime = event.timestamp
        init_msg_utime = ALL_CHANNELS_AND_LCMT[
            event.channel].decode(event.data).utime
        event = log_file.seek_to_timestamp(init_lcm_utime + start_utime)
        event = log_file.read_next_event()
        t_lcm_init = (event.timestamp - init_lcm_utime)*1e-6
        t_msg_init = (ALL_CHANNELS_AND_LCMT[
            event.channel].decode(event.data).utime - init_msg_utime)*1e-6

        lcm_t_of_last_goal_change = 0
        msg_t_of_last_goal_change = 0
        experiment_started = False
        goals_achieved = 0
        last_goal = None

        channels_and_n_msgs = {}

        while event is not None:
            if event.timestamp - init_lcm_utime > end_utime:
                break

            if event.channel not in channels_and_n_msgs.keys():
                channels_and_n_msgs[event.channel] = 1
            else:
                channels_and_n_msgs[event.channel] += 1

            if event.channel in self._channels:
                try:
                    msg_contents = ALL_CHANNELS_AND_LCMT[event.channel].decode(
                        event.data)
                except ValueError:
                    print(f'Failed to decode message from {event.channel}.')
                    breakpoint()
                lcm_secs = (event.timestamp - init_lcm_utime)*1e-6 - t_lcm_init
                msg_utime = msg_contents.utime
                msg_secs = (msg_utime - init_msg_utime)*1e-6 - t_msg_init

                # Cut off initial teleop -- detect goals based on the debug
                # message or the goal message.
                if self._use_debug_instead_of_target and \
                    event.channel == 'SAMPLING_C3_DEBUG':
                    if (not experiment_started) and \
                       (not msg_contents.is_teleop or not self.trim_bookends):
                        experiment_started = True
                        lcm_start_t = lcm_secs
                        msg_start_t = msg_secs
                        goals_achieved = msg_contents.detected_goal_changes
                    if experiment_started and \
                       (msg_contents.detected_goal_changes > goals_achieved):
                        goals_achieved = msg_contents.detected_goal_changes
                        lcm_t_of_last_goal_change = lcm_secs - lcm_start_t + \
                            self.lcm_t_adj
                        msg_t_of_last_goal_change = msg_secs - msg_start_t + \
                            self.msg_t_adj
                        print(f'Goal {goals_achieved} achieved at ' + \
                              f'{lcm_t_of_last_goal_change:.2f} s (' + \
                              f'{msg_t_of_last_goal_change:.2f} s from msg).')

                elif not self._use_debug_instead_of_target and \
                    event.channel == 'C3_FINAL_TARGET':
                    if not experiment_started:
                        experiment_started = True
                        lcm_start_t = lcm_secs
                        msg_start_t = msg_secs
                        last_goal = np.array(msg_contents.state[3:10])
                    new_goal = np.array(msg_contents.state[3:10])
                    if experiment_started and \
                        (np.linalg.norm(new_goal - last_goal) >= 1e-3):
                        goals_achieved += 1
                        last_goal = new_goal
                        if (msg_secs - msg_start_t + self.msg_t_adj) < \
                            msg_t_of_last_goal_change:
                            print(f'Found erroneously stitched experiments,' + \
                                  f' breaking.')
                            break
                        lcm_t_of_last_goal_change = lcm_secs - lcm_start_t + \
                            self.lcm_t_adj
                        msg_t_of_last_goal_change = msg_secs - msg_start_t + \
                            self.msg_t_adj
                        print(f'Goal {goals_achieved} achieved at ' + \
                              f'{lcm_t_of_last_goal_change:.2f} s (' + \
                              f'{msg_t_of_last_goal_change:.2f} s from msg).')

                if experiment_started:
                    self.messages_by_channel[event.channel][LCM_TIME_KEY
                        ].append(lcm_secs - lcm_start_t + self.lcm_t_adj)
                    self.messages_by_channel[event.channel][MESSAGE_KEY].append(
                        msg_contents)
                    try:
                        self.messages_by_channel[event.channel][MESSAGE_TIME_KEY
                            ].append(msg_secs - msg_start_t + self.msg_t_adj)
                    except:
                        pass

            event = log_file.read_next_event()

        self.goals_per_log.append(goals_achieved)

        # Before trimming:
        print(f'Channels and messages before trimming:')
        for key, val in channels_and_n_msgs.items():
            print(f'\t{key}: {val} messages')

        # Cut off the last goal since it was not achieved.
        if self.trim_bookends:
            trim_channel = 'SAMPLING_C3_DEBUG' if \
                self._use_debug_instead_of_target else 'C3_FINAL_TARGET'
            i_cutoff = self.messages_by_channel[trim_channel][
                LCM_TIME_KEY].index(lcm_t_of_last_goal_change)
            self.messages_by_channel[trim_channel][LCM_TIME_KEY] = \
                self.messages_by_channel[trim_channel][LCM_TIME_KEY][:i_cutoff]
            self.messages_by_channel[trim_channel][MESSAGE_TIME_KEY] = \
                self.messages_by_channel[trim_channel][MESSAGE_TIME_KEY][
                    :i_cutoff]
            self.messages_by_channel[trim_channel][MESSAGE_KEY] = \
                self.messages_by_channel[trim_channel][MESSAGE_KEY][:i_cutoff]
            self.lcm_t_adj = lcm_t_of_last_goal_change
            self.msg_t_adj = msg_t_of_last_goal_change

        if verbose:
            for channel, contents in self.messages_by_channel.items():
                print(f'Channel: {channel}')
                print(f'\tNum messages: {len(contents[LCM_TIME_KEY])}',
                      end = ', ')
                print(f'Time range: {contents[LCM_TIME_KEY][0]:.2f} to ' + \
                    f'{contents[LCM_TIME_KEY][-1]:.2f}')
            print(f'\nFinished processing log file at {log_filepath}.\n')

    def _downsample_channels(self, sync_time_key: str):
        """No need to keep more than 60Hz of information, so ensure every
        channel's messages are no more frequent than that."""
        for channel_name in self._channels:
            full_lcm_times = self.messages_by_channel[channel_name][
                LCM_TIME_KEY].copy()
            full_msg_times = self.messages_by_channel[channel_name][
                MESSAGE_TIME_KEY].copy()
            full_msgs = self.messages_by_channel[channel_name][
                MESSAGE_KEY].copy()
            print(f'Downsampling the "{channel_name}" channel to <=60Hz:  ' + \
                f'{len(full_lcm_times)} to ', end='')

            new_lcm_times = []
            new_msg_times = []
            new_msgs = []
            last_t = -1
            for i in range(len(full_lcm_times)):
                new_t = self.messages_by_channel[channel_name][sync_time_key][i]
                if new_t - last_t > (1.0/60):
                    last_t = new_t
                    new_lcm_times.append(full_lcm_times[i])
                    new_msg_times.append(full_msg_times[i])
                    new_msgs.append(full_msgs[i])

            self.messages_by_channel[channel_name][LCM_TIME_KEY] = new_lcm_times
            self.messages_by_channel[channel_name][MESSAGE_TIME_KEY] = \
                new_msg_times
            self.messages_by_channel[channel_name][MESSAGE_KEY] = new_msgs

            retention_percentage = 100*len(new_lcm_times)/len((full_lcm_times))
            print(f'{len(new_lcm_times)} ({retention_percentage:.2f}%)')

    def _synchronize_messages(self, sync_channels: List[str] = None,
                              synchronize_to_channel: str = 'SAMPLING_C3_DEBUG',
                              use_lcm_times: bool = True,
                              visualize: bool = True):
        """Synchronize the messages in the sync_channels list so their times
        match to within TIME_SYNC_THRESH of every message in the
        synchronize_to_channel.  If any channel in the sync_channels list cannot
        be synchronized at a given time, the time is discarded.  The result is
        all channels in sync_channels have the same number of messages, and each
        index corresponds across channels."""
        sync_channels = sync_channels if sync_channels is not None else \
            self._channels
        synchronize_to_channel = 'SAMPLING_C3_DEBUG' if 'SAMPLING_C3_DEBUG' in \
            self._channels else 'C3_FINAL_TARGET'
        time_key = LCM_TIME_KEY if use_lcm_times else MESSAGE_TIME_KEY

        # First, downsample the synchronize to channel if it is very high rate.
        self._downsample_channels(time_key)

        # Detect which channel had the fewest messages.
        min_num_channels = np.inf
        max_num_channels = 0
        min_channel = None
        max_channel = None
        for channel in sync_channels:
            n_msgs = len(self.messages_by_channel[channel][time_key])
            if n_msgs < min_num_channels:
                min_num_channels = n_msgs
                min_channel = channel
            if n_msgs > max_num_channels:
                max_num_channels = n_msgs
                max_channel = channel
        print(f'{min_channel} had fewest messages at {min_num_channels}')
        print(f'{max_channel} had most messages at {max_num_channels}')
        print(f'Synchronizing...', end=' ', flush=True)

        if visualize:
            _fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 1]})
            axs[0,0].sharey(axs[0,1])
            for channel in sync_channels:
                axs[0,0].plot(self.messages_by_channel[channel][time_key],
                              marker='o', label=channel)
            axs[0,0].set_title('All messages')
            axs[0,0].set_xlabel('Index')
            axs[0,0].set_ylabel('Time (s)')
            axs[0,0].legend()

            axs[1,0].plot(
                np.array(self.messages_by_channel[min_channel][time_key]) - \
                np.array(self.messages_by_channel[max_channel][time_key][
                    :min_num_channels]))
            axs[1,0].set_title(f'Time difference {min_channel} to ' + \
                               f'{max_channel}')
            axs[1,0].set_xlabel('Index')
            axs[1,0].set_ylabel('Time difference (s)')

        # Detect timestamps that are shared between all channels.
        ts = self.messages_by_channel[synchronize_to_channel][time_key].copy()
        problem_ts = []
        channel_problem_ts = []
        for channel in sync_channels:
            if channel == synchronize_to_channel:
                continue
            channel_ts = np.array(self.messages_by_channel[channel][time_key])
            for t in ts:
                delta = np.min(np.abs(channel_ts - t))
                if delta > TIME_SYNCH_THRESH and t not in problem_ts:
                    problem_ts.append(t)
                    channel_problem_ts.append(channel)
        for problem_t in problem_ts:
            ts.remove(problem_t)

        # Do some time synchronization to the minimum set of messages.
        for channel in sync_channels:
            new_lcm_times = []
            new_msg_times = []
            new_msgs = []
            channel_ts = np.array(self.messages_by_channel[channel][time_key])

            for t in ts:
                i = np.argmin(np.abs(channel_ts - t))
                new_lcm_times.append(self.messages_by_channel[channel][
                    LCM_TIME_KEY][i])
                new_msg_times.append(self.messages_by_channel[channel][
                    MESSAGE_TIME_KEY][i])
                new_msgs.append(
                    self.messages_by_channel[channel][MESSAGE_KEY][i])

            self.messages_by_channel[channel][LCM_TIME_KEY] = new_lcm_times
            self.messages_by_channel[channel][MESSAGE_TIME_KEY] = new_msg_times
            self.messages_by_channel[channel][MESSAGE_KEY] = new_msgs

        print(f'Done.')

        if visualize:
            for channel in self._channels:
                axs[0,1].plot(
                    self.messages_by_channel[channel][time_key], marker='o',
                    label=channel)
            axs[0,1].set_title('Synchronized messages')
            axs[0,1].set_xlabel('Index')
            axs[0,1].legend()

            axs[1,1].plot(
                np.array(self.messages_by_channel[min_channel][time_key]) - \
                np.array(self.messages_by_channel[max_channel][time_key]))
            axs[1,1].set_xlabel('Index')
            axs[1,1].set_ylabel('Time difference (s)')
            save_current_figure('time_sync')

    def _extract_information(self):
        if hasattr(self, 'times'):
            return
        if self._use_debug_instead_of_target:
            self._extract_information_with_debug()
        else:
            self._extract_information_without_debug()

    def _extract_information_without_debug(self):
        """Extracts and stores the following as numpy class attributes, if not
        done already, per time step:
            - times (N,)
            - pos_errors (N,)
            - rad_errors (N,)
            - n_goals_achieved (N,)
            - jack_poses (N, 7)
            - ee_locations (N, 3)

        And the following per goal:
            - times_of_new_goals (M,)
            - goals (M, 7)
            - worst_pos_errors (M,)
            - worst_rad_errors (M,)
        """
        # Get relevant messages over time.
        goals = self.messages_by_channel['C3_FINAL_TARGET'][MESSAGE_KEY]
        actuals = self.messages_by_channel['C3_ACTUAL'][MESSAGE_KEY]

        self.times = np.array(
            self.messages_by_channel['C3_FINAL_TARGET'][MESSAGE_TIME_KEY])

        # Things to keep track of for every timestamp.
        n_goals_achieved = []
        pos_errors, rad_errors = [], []
        ee_locations, jack_poses, target_poses = [], [], []

        # Things to keep track of for every goal.
        last_goal_num, last_goal = -1, np.zeros((7))
        times_of_new_goals, goal_poses = [], []
        worst_pos_errors, worst_rad_errors = [], []
        init_pos_errors, init_rad_errors = [], []

        for goal, actual, t in zip(goals, actuals, self.times):
            ee_locations.append(np.array(actual.state[:3]))
            jack_poses.append(np.array(actual.state[3:10]))
            target_poses.append(np.array(goal.state[3:10]))

            pos_errors.append(
                np.linalg.norm(jack_poses[-1][4:7] - target_poses[-1][4:7]))
            rad_errors.append(relative_angle_from_quats(
                jack_poses[-1][:4], target_poses[-1][:4]))

            if np.any(last_goal != np.array(goal.state[3:10])):
                goal_poses.append(np.array(goal.state[3:10]))
                last_goal = np.array(goal.state[3:10])
                last_goal_num += 1
                times_of_new_goals.append(t)
                init_pos_errors.append(pos_errors[-1])
                init_rad_errors.append(rad_errors[-1])
                worst_pos_errors.append(pos_errors[-1])
                worst_rad_errors.append(rad_errors[-1])

            n_goals_achieved.append(last_goal_num)

            if pos_errors[-1] > worst_pos_errors[-1]:
                worst_pos_errors[-1] = pos_errors[-1]
            if rad_errors[-1] > worst_rad_errors[-1]:
                worst_rad_errors[-1] = rad_errors[-1]

        self.n_goals_achieved = np.array(n_goals_achieved)
        self.pos_errors = np.array(pos_errors)
        self.rad_errors = np.array(rad_errors)
        self.ee_locations = np.array(ee_locations)
        self.jack_poses = np.array(jack_poses)

        self.times_of_new_goals = np.array(times_of_new_goals)
        self.goal_poses = np.array(goal_poses)
        self.worst_pos_errors = np.array(worst_pos_errors)
        self.worst_rad_errors = np.array(worst_rad_errors)

    def _extract_information_with_debug(self):
        """Extracts and stores the following as numpy class attributes, if not
        done already, per time step:
            - times (N,)
            - pos_errors (N,)
            - rad_errors (N,)
            - is_c3_mode_flags (N,)
            - n_goals_achieved (N,)
            - n_in_buffers (N,)
            - switch_reasons (N,)
            - jack_poses (N, 7)
            - ee_locations (N, 3)

        And the following per goal:
            - times_of_new_goals (M,)
            - goals (M, 7)
            - worst_pos_errors (M,)
            - worst_rad_errors (M,)
        """
        # Get relevant messages over time.
        sample_buffers = self.messages_by_channel['SAMPLE_BUFFER'][MESSAGE_KEY]
        debugs = self.messages_by_channel['SAMPLING_C3_DEBUG'][MESSAGE_KEY]
        goals = self.messages_by_channel['C3_FINAL_TARGET'][MESSAGE_KEY]
        actuals = self.messages_by_channel['C3_ACTUAL'][MESSAGE_KEY]

        self.times = np.array(
            self.messages_by_channel['C3_FINAL_TARGET'][MESSAGE_TIME_KEY])

        # Things to keep track of for every timestamp.
        n_in_buffers, n_goals_achieved, n_since_last_progress = [], [], []
        is_c3_mode_flags, pose_tracking = [], []
        pos_errors, rad_errors = [], []
        target_came_froms, switch_reasons = [], []
        ee_locations, jack_poses = [], []

        # Things to keep track of for every goal.
        last_goal_num, last_goal = -1, np.zeros((7))
        times_of_new_goals, goal_poses = [], []
        worst_pos_errors, worst_rad_errors = [], []
        init_pos_errors, init_rad_errors = [], []

        for buffer, debug, goal, actual, t in zip(
            sample_buffers, debugs, goals, actuals, self.times):

            n_in_buffers.append(buffer.num_in_buffer)
            n_goals_achieved.append(debug.detected_goal_changes)
            n_since_last_progress.append(debug.best_progress_steps_ago)

            is_c3_mode_flags.append(debug.is_c3_mode)
            pose_tracking.append(debug.in_pose_tracking_mode)

            pos_errors.append(debug.current_pos_error)
            rad_errors.append(debug.current_rot_error)

            target_came_froms.append(debug.source_of_pursued_target)
            switch_reasons.append(debug.mode_switch_reason)

            if last_goal_num != debug.detected_goal_changes:
                last_goal_num = debug.detected_goal_changes
                times_of_new_goals.append(t)
                init_pos_errors.append(debug.current_pos_error)
                init_rad_errors.append(debug.current_rot_error)
                worst_pos_errors.append(debug.current_pos_error)
                worst_rad_errors.append(debug.current_rot_error)

            if debug.current_pos_error > worst_pos_errors[-1]:
                worst_pos_errors[-1] = debug.current_pos_error
            if debug.current_rot_error > worst_rad_errors[-1]:
                worst_rad_errors[-1] = debug.current_rot_error

            if np.any(last_goal != np.array(goal.state[3:10])):
                goal_poses.append(np.array(goal.state[3:10]))
                last_goal = np.array(goal.state[3:10])

            ee_locations.append(np.array(actual.state[:3]))
            jack_poses.append(np.array(actual.state[3:10]))

        self.is_c3_mode_flags = np.array(is_c3_mode_flags)
        self.n_in_buffers = np.array(n_in_buffers)
        self.n_goals_achieved = np.array(n_goals_achieved)
        self.switch_reasons = np.array(switch_reasons)
        self.pos_errors = np.array(pos_errors)
        self.rad_errors = np.array(rad_errors)
        self.ee_locations = np.array(ee_locations)
        self.jack_poses = np.array(jack_poses)

        self.times_of_new_goals = np.array(times_of_new_goals)
        self.goal_poses = np.array(goal_poses)
        self.worst_pos_errors = np.array(worst_pos_errors)
        self.worst_rad_errors = np.array(worst_rad_errors)

        # Correct the goals achieved since they need to be cumulative across all
        # logs, and store how many goals per log.
        for log_i, downhill_i in enumerate(
            np.where(np.diff(self.n_goals_achieved) < 0)[0]):
            self.n_goals_achieved[downhill_i+1:] += self.goals_per_log[log_i]

    def _compute_times_to_thresholds(self):
        """Per goal, computes the time the controller took to get the object to
        the goal within each set of thresholds.  Creates the class attribute
        self.times_to_thresholds of shape (n_goals, n_thresholds), with the
        first column as the tightest threshold."""
        if hasattr(self, 'times_to_thresholds'):
            return

        self._get_trajectory_tolerances()

        # Use all tolerances up to as fine as the used one from the log.
        pos_thresholds = POS_SUCCESS_THRESHOLDS[
            POS_SUCCESS_THRESHOLDS >= self.pos_tol]
        rad_thresholds = RAD_SUCCESS_THRESHOLDS[
            RAD_SUCCESS_THRESHOLDS >= self.rad_tol]

        # Compute the time it took to reach each threshold level for each goal.
        n_goals = self.n_goals_achieved[-1] + 1
        times_to_thresholds = np.zeros((n_goals, len(pos_thresholds)))
        for goal_i, goal_t in enumerate(self.times_of_new_goals):
            for thresh_i, (pos_thresh, rad_thresh) in enumerate(
                zip(pos_thresholds, rad_thresholds)):

                if pos_thresh == self.pos_tol and rad_thresh == self.rad_tol:
                    if goal_i == n_goals - 1:
                        times_to_thresholds[goal_i, thresh_i] = \
                            self.times[-1] - goal_t
                    else:
                        times_to_thresholds[goal_i, thresh_i] = \
                            self.times_of_new_goals[goal_i+1] - goal_t
                    continue

                time_i = np.argmin(
                    np.abs(self.times - self.times_of_new_goals[goal_i])) + 1
                already_set = False
                while (self.pos_errors[time_i] > pos_thresh.item()) or \
                    (self.rad_errors[time_i] > rad_thresh.item()):
                    time_i += 1
                    if time_i == len(self.times):
                        assert thresh_i > 0
                        times_to_thresholds[goal_i, thresh_i] = \
                            times_to_thresholds[goal_i, thresh_i-1]
                        already_set = True
                        break
                if not already_set:
                    times_to_thresholds[goal_i, thresh_i] = \
                        self.times[time_i] - goal_t

        self.times_to_thresholds = times_to_thresholds

    def _flag_hardware_violations(self):
        """Detects hardware violations in the recorded logs.  For hardware
        experiments, this is unnecessary.  For simulation experiments, this can
        catch results that would not have been possible on hardware due to
        safety and robot limits.  This includes:
            - joint limits
            - joint velocity limits
            - joint torque limits
            - C3 workspace limits.
        In addition, this method approximates the joint acceleration and jerk by
        finite differencing the reported joint velocities.  This approximation
        seems to be noisy enough to trigger false positves (e.g. they can be
        triggered even on hardware logs), so these are only computed for
        reference and not reported as a violation.

        Stores information in the following attributes:
            - goal_violations:  dict with keys for joint, velocity, torque, and
                workspace limits, with values of the goal numbers which were in
                violation.
            - bad_t_states:  (?, 8) with t followed by bad joint positions
            - bad_t_vels:  (?, 8) with t followed by bad joint velocities
            - bad_t_accs:  (?, 8) with t followed by bad joint accelerations
            - bad_t_jerks:  (?, 8) with t followed by bad joint jerks
            - bad_t_torques:  (?, 8) with t followed by bad joint torques
            - bad_t_ee_xyzs:  (?, 4) with t followed by bad EE xyz locations
        """
        if hasattr(self, 'goal_violations'):
            return

        self._extract_information()
        self._detect_workspace_violations()

        goal_violations = {}
        keys = [JOINT_LIMIT_VIO_KEY, JOINT_VEL_VIO_KEY, JOINT_TORQUE_VIO_KEY,
                WSL_VIO_KEY]
        for key in keys:  goal_violations[key] = []
        for t_thing in self.bad_t_ee_xyzs:
            t = t_thing[0]
            goal_num = int(np.where(t > self.times_of_new_goals)[0][-1]) + 1
            if goal_num not in goal_violations[WSL_VIO_KEY]:
                goal_violations[WSL_VIO_KEY].append(goal_num)

        start_utime = int(0)

        # Since Franka state information is very high frequency, only store the
        # problematic ones.
        bad_t_states = np.zeros((0, 8))
        bad_t_vels = np.zeros((0, 8))
        bad_t_accs = np.zeros((0, 8))
        bad_t_jerks = np.zeros((0, 8))
        bad_t_torques = np.zeros((0, 8))

        # Iterate per log.
        # goal_adjs = [0] + [int(num) for num in self.goals_per_log[:-1]]
        for i, log_filepath in enumerate(self.log_filepaths):

            # Get the right end time in the log to exclude any non-achieved
            # goals at the end.
            start_time_adj = self.log_lcm_start_times[i]
            end_time = self.log_lcm_start_times[i+1]-self.log_lcm_start_times[i]
            end_utime = int(end_time*1e6)

            # Need to keep up with the past 3 velocity reports for computing
            # acceleration and jerk violations.
            previous_3_t_vels = np.zeros((0, 8))

            # Open the LCM log.
            log_file = EventLog(log_filepath, 'r')

            # Read through the log file.
            event = log_file.read_next_event()
            while event.channel not in self._channels:
                event = log_file.read_next_event()
            init_lcm_utime = event.timestamp
            init_msg_utime = ALL_CHANNELS_AND_LCMT[
                event.channel].decode(event.data).utime
            event = log_file.seek_to_timestamp(init_lcm_utime + start_utime)
            event = log_file.read_next_event()
            t_msg_init = (ALL_CHANNELS_AND_LCMT[
                event.channel].decode(event.data).utime - init_msg_utime)*1e-6

            print(f'Reading Franka states from {log_filepath}... ', end='',
                  flush=True)
            while event is not None:
                if event.timestamp - init_lcm_utime > end_utime:
                    break

                if event.channel in ['FRANKA_STATE', 'FRANKA_STATE_SIMULATION']:
                    try:
                        msg_contents = ALL_CHANNELS_AND_LCMT[
                            event.channel].decode(event.data)
                    except ValueError:
                        print(f'Failed to decode message from {event.channel}.')
                        breakpoint()
                    msg_utime = msg_contents.utime
                    msg_secs = (msg_utime - init_msg_utime)*1e-6 - \
                        t_msg_init + start_time_adj

                    # Compute the states.
                    t_pos = np.hstack((
                        np.array([msg_secs]), np.array(msg_contents.position)))
                    t_vel = np.hstack((
                        np.array([msg_secs]), np.array(msg_contents.velocity)))
                    t_torque = np.hstack((
                        np.array([msg_secs]), np.array(msg_contents.effort)))

                    # Maintain the past 3 velocities.
                    if previous_3_t_vels.shape[0] < 3:
                        previous_3_t_vels = np.concatenate((
                            previous_3_t_vels, t_vel.reshape(1, 8)))
                    else:
                        previous_3_t_vels[0, :] = previous_3_t_vels[1, :]
                        previous_3_t_vels[1, :] = previous_3_t_vels[2, :]
                        previous_3_t_vels[2, :] = t_vel

                    # Store any problematic ones.
                    if np.any(t_pos[1:] < FRANKA_JOINT_MINS) or \
                    np.any(t_pos[1:] > FRANKA_JOINT_MAXS):
                        bad_t_states = np.concatenate((
                            bad_t_states, t_pos.reshape(1, 8)))
                    if np.any(np.abs(t_vel[1:]) > FRANKA_JOINT_VEL_LIMITS):
                        bad_t_vels = np.concatenate((
                            bad_t_vels, t_vel.reshape(1, 8)))
                    if np.any(np.abs(t_torque[1:])>FRANKA_JOINT_TORQUE_LIMITS):
                        bad_t_torques = np.concatenate((
                            bad_t_torques, t_torque.reshape(1, 8)))

                    if previous_3_t_vels.shape[0] >= 2:
                        dt = previous_3_t_vels[-1, 0] - previous_3_t_vels[-2, 0]
                        acc = (previous_3_t_vels[-1, 1:] - \
                            previous_3_t_vels[-2, 1:])/dt
                        t_acc = np.hstack((np.array([msg_secs]), acc))
                        if np.any(np.abs(t_acc[1:]) > FRANKA_JOINT_ACC_LIMITS):
                            bad_t_accs = np.concatenate((
                                bad_t_accs, t_acc.reshape(1, 8)))
                    if previous_3_t_vels.shape[0] == 3:
                        prev_dt = previous_3_t_vels[-2, 0] - \
                            previous_3_t_vels[-3, 0]
                        prev_acc = (previous_3_t_vels[-2, 1:] - \
                                    previous_3_t_vels[-3, 1:])/prev_dt
                        jerk = (acc - prev_acc)/dt
                        t_jerk = np.hstack((np.array([msg_secs]), jerk))
                        if np.any(np.abs(t_jerk[1:])>FRANKA_JOINT_JERK_LIMITS):
                            bad_t_jerks = np.concatenate((
                                bad_t_jerks, t_jerk.reshape(1, 8)))

                event = log_file.read_next_event()
            print(f'Done.\n')

        print(f'\nJoint limit violations: {bad_t_states.shape[0]}')
        print(f'Joint velocity violations: {bad_t_vels.shape[0]}')
        print(f'Joint acceleration violations: {bad_t_accs.shape[0]}')
        print(f'Joint jerk violations: {bad_t_jerks.shape[0]}')
        print(f'Joint torque violations: {bad_t_torques.shape[0]}')
        print(f'WSL violations: {self.bad_t_ee_xyzs.shape[0]}')

        t_somethings = [bad_t_states, bad_t_vels, bad_t_torques]
        for key, t_something in zip(keys, t_somethings):
            goal_violations[key] = []
            for t_thing in t_something:
                t = t_thing[0]
                goal_num = int(np.where(t > self.times_of_new_goals)[0][-1]) + 1
                if goal_num not in goal_violations[key]:
                    goal_violations[key].append(goal_num)

        self.goal_violations = goal_violations
        print(f'\nDetected hardware violations per goal:')
        for key, val in goal_violations.items():
            print(f'\t{key}: {val}')

        self.bad_t_states = bad_t_states
        self.bad_t_vels = bad_t_vels
        self.bad_t_accs = bad_t_accs
        self.bad_t_jerks = bad_t_jerks
        self.bad_t_torques = bad_t_torques

    def _detect_workspace_violations(self):
        """Detect any violation of workspace limits throughout the experiment.
        """
        self._get_workspace_limits()

        problem_t_ee_xyzs = np.zeros((0, 4))

        # Iterate over every C3_ACTUAL message.
        for t, ee_xyz in zip(self.times, self.ee_locations):
            x, y, z = ee_xyz
            radius = np.linalg.norm(ee_xyz)
            x, y, z, radius = float(x), float(y), float(z), float(radius)
            if (x < self.wsl_x[0] or x > self.wsl_x[1]) or \
               (y < self.wsl_y[0] or y > self.wsl_y[1]) or \
               (z < self.wsl_z[0] or z > self.wsl_z[1]) or \
               (radius < self.wsl_r[0] or radius > self.wsl_r[1]):
                problem_t_ee_xyzs = np.concatenate((
                    problem_t_ee_xyzs, np.array([[t, x, y, z]])))

        self.bad_t_ee_xyzs = problem_t_ee_xyzs

    def prepare_and_export(self, filename: str):
        """Perform all of the compute-heavy analysis and export the class as a
        pickle object.  This can be reloaded in the future.  Note:  Since the
        special LCM type objects are not pickle-able, need to remove the
        self.messages_by_channel attribute.  This is ok because all of the
        computations that require this information are already done before
        exporting; all future visualizations are still runnable."""
        self._extract_information()
        self._flag_hardware_violations()
        self._compute_times_to_thresholds()

        del self.__dict__['messages_by_channel']
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f'\nWrote ResultsAnalyzer class as pickle object at:  {filename}')

    def inspect_mode_switching_by_goal(self):
        """Generate mode switching plots for every goal achieved.  The plots
        show the position and orientation errors over time, with shading for
        contact-rich and contact-free modes, with colored vertical separation
        lines indicating the reasons for the mode switching."""
        self._extract_information()
        self._get_trajectory_tolerances()

        n_goals = self.n_goals_achieved[-1] + 1
        for i in range(n_goals):
            ts = self.times[self.n_goals_achieved == i]
            is_c3s = None if not self._use_debug_instead_of_target else \
                self.is_c3_mode_flags[self.n_goals_achieved == i]
            switches = None if not self._use_debug_instead_of_target else \
                self.switch_reasons[self.n_goals_achieved == i]
            pos_es = self.pos_errors[self.n_goals_achieved == i]
            rad_es = self.rad_errors[self.n_goals_achieved == i]
            inspect_mode_switching_by_goal(
                ts, is_c3s, switches, pos_es, rad_es, self.pos_tol,
                self.rad_tol, i+1, log_folder=self.save_folder)

    def visualize_goals_with_violations(self, title_suffix: str = None):
        """Make a plot of the time to reach each goal, ordered from shortest to
        longest, where each bar is colored to denote whether a hardware limit
        violation occurred in pursuit of the goal."""
        self._extract_information()
        self._flag_hardware_violations()
        self._compute_times_to_thresholds()

        title = 'Time to Reach Goal'
        title += f': {title_suffix}' if title_suffix is not None else ''

        # Use the tightest thresholds.
        goal_times = self.times_to_thresholds[:, 0]

        # Differentiate the problem times from the good times.
        problem_goals = []
        problem_goals += self.goal_violations[JOINT_LIMIT_VIO_KEY]
        problem_goals += self.goal_violations[JOINT_VEL_VIO_KEY]
        problem_goals += self.goal_violations[JOINT_TORQUE_VIO_KEY]
        valid_goal_mask = np.array([i not in problem_goals for i in range(
            1, self.n_goals_achieved[-1].item() + 2)])
        invalid_fraction = (~valid_goal_mask).sum()/valid_goal_mask.shape[0]

        # Sort based on shortest to longest time to goal.
        sorted_idx = goal_times.argsort()
        sorted_times = goal_times[sorted_idx]
        sorted_valid_goal_mask = valid_goal_mask[sorted_idx]

        xs = np.arange(1, len(goal_times) + 1)

        average_time = np.mean(sorted_times)

        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        axs.bar(xs[sorted_valid_goal_mask],
                sorted_times[sorted_valid_goal_mask], width=1, label='Valid')
        axs.bar(xs[~sorted_valid_goal_mask],
                sorted_times[~sorted_valid_goal_mask], width=1,
                label='Violated Franka Limits')
        axs.hlines(average_time, 0.5, len(goal_times)+0.5, linestyles='--',
                   label=f'Average: {average_time:.2f}s')
        axs.set_xlabel(f'Goals ({100*invalid_fraction:.1f}% in violation of' + \
                       f' hardware limits)', fontsize = 16)
        axs.set_ylabel('Time [s]', fontsize = 16)
        axs.set_title(title, fontsize = 18)
        axs.set_xlim([0.5, len(goal_times)+0.5])
        axs.set_ylim([0, 400])
        axs.set_xticks(xs)
        axs.legend(fontsize=14)
        save_current_figure(f'goal_violations_{title_suffix.replace(" ", "_")}',
                            store_folder=self.save_folder)

    def visualize_time_histograms(self):
        """Generate time-to-goal histograms."""
        self._extract_information()
        self._compute_times_to_thresholds()

        # Use all tolerances up to as fine as the used one from the log.
        pos_thresholds = POS_SUCCESS_THRESHOLDS[
            POS_SUCCESS_THRESHOLDS >= self.pos_tol]
        rad_thresholds = RAD_SUCCESS_THRESHOLDS[
            RAD_SUCCESS_THRESHOLDS >= self.rad_tol]
        colors = THRESHOLD_COLORS[-len(pos_thresholds):]

        # Compute the time it took to reach each threshold level for each goal.
        times_to_thresholds = self.times_to_thresholds

        # Generate a plot of time histograms per threshold.
        fig, axs = plt.subplot_mosaic(
            [[0, 4], [1, 4], [2, 4], [3, 4]], constrained_layout=True,
            figsize=(9, 9), sharex=True, sharey=False)
        axs[1].sharey(axs[0])
        axs[2].sharey(axs[0])
        axs[3].sharey(axs[0])
        fig.suptitle('Time to Reach Goal')
        handles = []

        counts, bins = np.histogram(times_to_thresholds, bins=HIST_BINS,
                                    range=(0, np.max(times_to_thresholds)))

        for thresh_i, color in enumerate(colors):
            data = times_to_thresholds[:, thresh_i]
            kde = gaussian_kde(data)
            x = np.linspace(min(data), max(data), 100)
            label=f'{pos_thresholds[thresh_i]:.2f}m, ' + \
                f'{rad_thresholds[thresh_i]:.1f}rad'
            axs[thresh_i].hist(data, color=color, alpha=0.5, density=True,
                            bins=bins, label=label)
            handles.append(
                axs[thresh_i].plot(
                    x, kde(x), color=color, linewidth=4, label=label)[0])
            axs[thresh_i].set_ylabel('Probability Density')
            axs[thresh_i].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            axs[4].hist(data, color=color, alpha=0.5, density=True, bins=bins)
            axs[4].plot(x, kde(x), color=color, linewidth=4, label=label)

        axs[3].set_xlabel('Time [s]')
        axs[4].set_xlabel('Time [s]')
        axs[4].set_ylabel('Probability Density')
        axs[4].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axs[4].legend(handles=handles, title='Thresholds')
        plt.tight_layout()
        save_current_figure('time_hist', store_folder=self.save_folder)

    def visualize_goal_success(self):
        """Generate a debugging-purposed goal success plot."""
        self._extract_information()
        self._compute_times_to_thresholds()

        # Use all tolerances up to as fine as the used one from the log.
        pos_thresholds = POS_SUCCESS_THRESHOLDS[
            POS_SUCCESS_THRESHOLDS >= self.pos_tol]
        rad_thresholds = RAD_SUCCESS_THRESHOLDS[
            RAD_SUCCESS_THRESHOLDS >= self.rad_tol]
        colors = THRESHOLD_COLORS[-len(pos_thresholds):]

        # Compute the time it took to reach each threshold level for each goal.
        times_to_thresholds = self.times_to_thresholds

        # Generate debugging plot for goal success.
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        axs[0].sharex(axs[1])
        axs[0].plot(self.times, self.pos_errors, label='Position error')
        for pos_thresh, color in zip(pos_thresholds, colors):
            axs[0].axhline(y=pos_thresh, linestyle='--', color=color,
                        label=f'{pos_thresh:.2f}m threshold')
        axs[1].plot(self.times, self.rad_errors, label='Rotation error')
        for rad_thresh, color in zip(rad_thresholds, colors):
            axs[1].axhline(y=rad_thresh, linestyle='--', color=color,
                        label=f'{rad_thresh:.1f}rad threshold')
        axs[0].set_xlabel('Time (s)', fontsize = 16)
        axs[1].set_xlabel('Time (s)', fontsize = 16)
        axs[0].set_ylabel('Error [m]', fontsize = 16)
        axs[1].set_ylabel('Error [rad]', fontsize = 16)
        axs[0].set_title('Position Error', fontsize = 18)
        axs[1].set_title('Orientation Error', fontsize = 18)
        axs[0].legend(fontsize=14, title_fontsize=16)
        axs[1].legend(fontsize=14, title_fontsize=16)
        for goal_i in range(times_to_thresholds.shape[0]):
            for thresh_i, color in enumerate(colors):
                time = times_to_thresholds[goal_i, thresh_i] + \
                    self.times_of_new_goals[goal_i]
                time_i = np.argmin(np.abs(self.times - time)).item()
                axs[0].axvline(x=time, color=color, alpha=0.5)
                axs[1].axvline(x=time, color=color, alpha=0.5)

        n_goals, n_thresholds = times_to_thresholds.shape
        x = np.arange(n_goals)
        bar_width = 0.2
        offsets = np.linspace(-bar_width * (n_thresholds - 1) / 2,
                              bar_width * (n_thresholds - 1) / 2,
                              n_thresholds)

        for thresh_i, color in enumerate(colors):
            axs[2].bar(x + offsets[thresh_i], times_to_thresholds[:, thresh_i],
                    width=0.2, color=color,
                    label=f'{pos_thresholds[thresh_i]:.2f}m, ' + \
                        f'{rad_thresholds[thresh_i]:.1f}rad')
        axs[2].set_ylabel('Time [s]', fontsize = 16)
        axs[2].set_title('Time to Reach Goal', fontsize = 18)
        axs[2].set_xticks(x)
        axs[2].set_xticklabels([f'Goal {i+1}' for i in range(n_goals)])
        axs[2].legend(title="Thresholds", bbox_to_anchor=(1.02, 1),
                      loc='upper left', fontsize=14, title_fontsize=16)
        plt.tight_layout()
        save_current_figure('goal_success', store_folder=self.save_folder)

    def visualize_time_to_goal_vs_error(self):
        """Generate time to goal versus orientation and position error plots."""
        self._extract_information()
        self._compute_times_to_thresholds()

        # Use all tolerances up to as fine as the used one from the log.
        pos_thresholds = POS_SUCCESS_THRESHOLDS[
            POS_SUCCESS_THRESHOLDS >= self.pos_tol]
        rad_thresholds = RAD_SUCCESS_THRESHOLDS[
            RAD_SUCCESS_THRESHOLDS >= self.rad_tol]
        colors = THRESHOLD_COLORS[-len(pos_thresholds):]

        # Compute the time it took to reach each threshold level for each goal.
        times_to_thresholds = self.times_to_thresholds

        # Set a general serif font (e.g., Times New Roman or alternative)
        plt.rcParams.update({'font.family': 'serif'})

        # Generate a plot of time to goal versus worst errors incurred over the
        # trajectory.
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for thresh_i, color in enumerate(colors):
            axs[0].scatter(
                self.worst_pos_errors, times_to_thresholds[:, thresh_i],
                color=color, label=f'{pos_thresholds[thresh_i]:.2f}' + \
                    f'm, {rad_thresholds[thresh_i]:.1f}rad')
            axs[1].scatter(
                self.worst_rad_errors*180/np.pi,
                times_to_thresholds[:, thresh_i], color=color,
                label=f'{pos_thresholds[thresh_i]:.2f}m, ' + \
                    f'{rad_thresholds[thresh_i]:.1f}rad')
        axs[1].legend(title='Success Thresholds', fontsize = 14, title_fontsize = 16)
        axs[0].set_xlabel('Position Error [m]', fontsize = 16)
        axs[1].set_xlabel('Orientation Error [deg]', fontsize = 16)
        axs[0].set_ylabel('Time to Goal [s]', fontsize = 16)
        fig.suptitle('Time to Goal vs. Worst Error Over Trajectory',  fontsize = 18)
        save_current_figure('time_vs_error', store_folder=self.save_folder)

    def visualize_cdf(self):
        """Generate CDF plot."""
        self._extract_information()
        self._compute_times_to_thresholds()

        # Use all tolerances up to as fine as the used one from the log.
        pos_thresholds = POS_SUCCESS_THRESHOLDS[
            POS_SUCCESS_THRESHOLDS >= self.pos_tol]
        rad_thresholds = RAD_SUCCESS_THRESHOLDS[
            RAD_SUCCESS_THRESHOLDS >= self.rad_tol]
        colors = THRESHOLD_COLORS[-len(pos_thresholds):]

        # Compute the time it took to reach each threshold level for each goal.
        times_to_thresholds = self.times_to_thresholds

        # Set a general serif font (e.g., Times New Roman or alternative)
        plt.rcParams.update({'font.family': 'serif'})

        # Generate a plot of cumulative distribution functions.
        fig, axs = plt.subplots(1, 1, figsize=(6,4))
        for thresh_i, color in enumerate(colors):
            data = times_to_thresholds[:, thresh_i]
            count, bins_count = np.histogram(
                data, bins=11, range=(0, 1.1*CDF_TIME_CUTOFF))
            pdf = count / sum(count)
            cdf = np.cumsum(pdf)
            bins_count[0] = 0
            cdf = np.concatenate([[0], cdf])
            axs.plot(bins_count, cdf, color=color, linewidth=5,
                     label=f'{pos_thresholds[thresh_i]:.2f}' + \
                        f'm, {rad_thresholds[thresh_i]:.1f}rad')
        axs.legend(title='Pose Success Thresholds', fontsize=14,
                   title_fontsize=16)
        axs.set_xlabel('Time Limit [s]', fontsize=16)
        axs.set_ylabel('Fraction of Trials', fontsize=16)
        axs.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axs.set_yticks(np.linspace(0, 1, 11))
        axs.set_ylim([0, 1])
        axs.set_xlim([0, CDF_TIME_CUTOFF])
        axs.tick_params(axis='both', which='major', labelsize=12)

        # Adjust margins to prevent labels from being cut off
        fig.subplots_adjust(left=0.15, bottom=0.15)

        plt.grid()
        fig.suptitle('Fraction of Goals Achieved Within Time Limit',
                     fontsize=18)
        save_current_figure('cdf', store_folder=self.save_folder)

    def print_trim_times(self):
        """Generate the ffmpeg commands to run to split the phone videos of the
        experiments into goal-length snippets.  Note, this does not run those
        commands, only prints them to terminal output so they can be copy-pasted
        into a terminal and run as needed."""
        assert len(self.log_filepaths) == 1, f'Only one log file can be ' + \
            f'used to print ffmpeg commands since each video is separated.'
        assert self.log_filepaths[0] in LOG_FILEPATHS_TO_VIDEOS.keys(), \
            f'Only prepared to print ffmpeg commands for log files in ' + \
            f'{LOG_FILEPATHS_TO_VIDEOS.keys()=} but got {self.log_filepaths[0]}'
        log_filepath = self.log_filepaths[0]

        input_video, output_dir = LOG_FILEPATHS_TO_VIDEOS[log_filepath]
        if not op.exists(output_dir):
            print(f'Making {output_dir}...')
            os.makedirs(output_dir)
        else:
            print(f'Found existing {output_dir}...')

        date, log_num = get_date_and_log_num_from_log_filepath(log_filepath)

        self._extract_information()
        self._compute_times_to_thresholds()

        cut_times = np.concatenate((self.times_of_new_goals, self.times[-1:]))
        n_goals = len(self.times_of_new_goals)
        assert cut_times.shape[0] == n_goals + 1

        for i in range(n_goals):
            t_start, t_end = cut_times[i], cut_times[i+1]
            h_start, h_end = int(t_start // 3600), int(t_end // 3600)
            rem_start, rem_end = t_start-h_start*3600, t_end-h_end*3600
            m_start, m_end = int(rem_start // 60), int(rem_end // 60)
            s_start, s_end = t_start % 60, t_end % 60

            output_video_name = f'{date}_log_{log_num}_goal_{i+1}.mp4'

            # Note: be careful with -y since this will overwrite things.
            print(f'ffmpeg -y -i {input_video} -ss {h_start}:{m_start}:' + \
                  f'{s_start:.3f} -to {h_end}:{m_end}:{s_end:.3f} ' + \
                  f'-vcodec libx264 {op.join(output_dir, output_video_name)}')

    def overlay_videos(self, per_goal: bool = True):
        """Prepare to generate overlay videos."""
        assert len(self.log_filepaths) == 1, f'Only one log file can be ' + \
            f'used to make overlay videos since each experiment is separated.'
        assert self.log_filepaths[0] in LOG_FILEPATHS_TO_VIDEOS.keys(), \
            f'Only prepared to print ffmpeg commands for log files in ' + \
            f'{LOG_FILEPATHS_TO_VIDEOS.keys()=} but got {self.log_filepaths[0]}'
        log_filepath = self.log_filepaths[0]

        input_video, output_dir = LOG_FILEPATHS_TO_VIDEOS[log_filepath]
        overlay_dir = op.join(output_dir, 'overlay')
        if not op.exists(overlay_dir):
            print(f'Making {overlay_dir}...')
            os.makedirs(overlay_dir)
        else:
            print(f'Found existing {overlay_dir}...')

        self._extract_information()
        self._compute_times_to_thresholds()

        if per_goal:
            self._per_goal_overlay(log_filepath, output_dir, overlay_dir)
        else:
            self._full_video_overlay(
                log_filepath, output_dir, overlay_dir, input_video)

    def _full_video_overlay(self, log_filepath: str, output_dir: str,
                            overlay_dir: str, input_video: str):
        """Make a continuous overlay video with text annotations to count the
        goals achieved."""
        print(f'Full video overlay')
        date, log_num = get_date_and_log_num_from_log_filepath(log_filepath)
        n_goals = len(self.times_of_new_goals)

        # First overlay the videos.
        full_meshcat_video = op.join(
            output_dir, f'goal_times_0_{int(self.times[-1])}.mp4')
        output_video = op.join(
            overlay_dir, f'overlay_{date}_log_{log_num}_full.mp4')
        placement = BOTTOM_RIGHT_PLACEMENT if log_is_push_t(log_filepath) \
            else BOTTOM_LEFT_PLACEMENT
        cmd = f'ffmpeg -y -i {input_video} -i {full_meshcat_video} ' + \
              f'-filter_complex "[1:v]scale=500:-1[v2];[0:v][v2]' + \
              f'overlay=main_w-overlay_w-{placement}" -c:v libx264 -c:a ' + \
              f'copy {output_video}'
        print(f'\n{cmd}\n')
        os.system(cmd)

        # Second add text annotations to count the goals achieved.
        goal_labeled_video = output_video.replace('.mp4', '_labeled.mp4')
        cmd = f'ffmpeg -y -i {output_video} -vf "'
        placement = UPPER_RIGHT_TEXT_PLACEMENT if log_is_push_t(log_filepath) \
            else UPPER_LEFT_TEXT_PLACEMENT
        for goal in range(n_goals+1):
            if goal == n_goals:
                t_start = self.times[-1]
                t_end = get_video_duration(input_video)
            else:
                t_start = self.times_of_new_goals[goal]
                t_end = self.times[-1] if goal==n_goals-1 else \
                    self.times_of_new_goals[goal+1]
            goal_txt = f'{goal} Goal' if goal==1 else f'{goal} Goals'
            cmd += f"drawtext=text='{goal_txt}':{placement}:fontsize=150:" + \
                   f"fontcolor_expr=0x00FFFFFF:enable='between(t," + \
                   f"{t_start:.3f},{t_end:.3f})',"
        placement = BOTTOM_RIGHT_BOX_PLACEMENT if log_is_push_t(log_filepath) \
            else BOTTOM_LEFT_BOX_PLACEMENT
        cmd += f"drawbox={placement}:w=680:h=395:color=gray@0.5:t=fill:" + \
               f"enable='between(t,{t_start:.3f},{t_end:3f})'"
        cmd += f'" -c:a copy {goal_labeled_video}'
        print(f'\n{cmd}\n')
        os.system(cmd)

    def _per_goal_overlay(self, log_filepath: str, output_dir: str,
                          overlay_dir: str):
        """Make a single overlay video per goal with end screen annotation to
        express the goal success."""
        date, log_num = get_date_and_log_num_from_log_filepath(log_filepath)
        n_goals = len(self.times_of_new_goals)

        for i in range(n_goals):
            phone_video = op.join(
                output_dir, f'{date}_log_{log_num}_goal_{i+1}.mp4')
            meshcat_video = op.join(output_dir, f'goal_goal_{i+1}.mp4')
            output_video = op.join(
                overlay_dir, f'overlay_{date}_log_{log_num}_goal_{i+1}.mp4')

            # First overlay the videos.
            placement = BOTTOM_RIGHT_PLACEMENT if log_is_push_t(log_filepath) \
                else BOTTOM_LEFT_PLACEMENT
            cmd = f'ffmpeg -y -i {phone_video} -i {meshcat_video} ' + \
                  f'-filter_complex "[1:v]scale=500:-1[v2];[0:v][v2]' + \
                  f'overlay=main_w-overlay_w-{placement}" -c:v libx264 -c:a' + \
                  f' copy {output_video}'
            print(f'\n{cmd}\n')
            os.system(cmd)

            # Second extract the last frame.
            last_frame = output_video.replace('.mp4', '_last_frame.jpg')
            cmd = f'ffmpeg -y -sseof -3 -i {output_video} -q:v 1 ' + \
                  f'-update 1 {last_frame}'
            print(f'\n{cmd}\n')
            os.system(cmd)

            # Third add rectangle to the last frame and make short video of it.
            shaded_last_frame = last_frame.replace('.jpg', '_shaded.jpg')
            tolerance_text = f'within {int(self.pos_tol*100)}cm and ' + \
                f'{self.rad_tol*180/np.pi:.1f}degrees'
            cmd = f'convert {last_frame} -fill "rgba(0,255,0,0.5)" ' + \
                  f'-stroke black -draw "rectangle 0,0 1920,1080" -fill ' + \
                  f'white -stroke black -pointsize 150 -draw "text 50,500 ' + \
                  f"'{tolerance_text}'" +  f'" {shaded_last_frame}'
            print('\n', cmd, '\n')
            os.system(cmd)
            still_video = shaded_last_frame.replace('.jpg', '.mp4')
            # FPS is different for iPhone videos.
            fps = '29.97' if '01_14_25' in self.log_filepaths[0] else '30'
            cmd = f'ffmpeg -loop 1 -i {shaded_last_frame} -t 2 -s ' + \
                  f'1920x1080 -r {fps} -pixel_format yuv420p -y {still_video}'
            print('\n', cmd, '\n')
            os.system(cmd)

            # A required intermediate for iPhone videos.
            if '01_14_25' in self.log_filepaths[0]:
                original_output_video = output_video
                output_video = output_video.replace('.mp4', '_29-97fps.mp4')
                cmd1 = f'ffmpeg -y -i {original_output_video} -c:v copy ' + \
                      f'-video_track_timescale 29.97 {output_video}'
                print(f'\n{cmd1}\n')
                os.system(cmd1)
                original_still_video = still_video
                still_video = still_video.replace('.mp4', '_29-97fps.mp4')
                cmd1 = f'ffmpeg -y -i {original_still_video} -c:v copy ' + \
                      f'-video_track_timescale 29.97 {still_video}'
                print(f'\n{cmd1}\n')
                os.system(cmd1)

            # Fourth add the still frame to the end of the video.
            success_video = output_video.replace('.mp4', '_success.mp4')
            cmd = f'ffmpeg -i {output_video} -i {still_video} ' + \
                   f'-filter_complex "[0:v:0][1:v:0]concat=n=2:v=1[outv]" ' + \
                   f'-map "[outv]" -y {success_video}'
            print('\n', cmd, '\n')
            os.system(cmd)

    def generate_goal_video(self, t_init: float = None, t_final: float = None,
                            per_goal: bool = False):
        self._extract_information()

        if per_goal:
            assert t_init == t_final == None, f'Not prepared to trim videos' + \
                f' if exporting videos per goal.'
            print(f'Generating multiple videos, one for each goal.')
            t_inits = self.times_of_new_goals
            t_finals = np.concatenate((t_inits[1:], self.times[-1:]))
        else:
            print(f'Generating one continuous video for the whole log.')
            t_inits = [0] if t_init is None else [t_init]
            t_finals = [self.times[-1]] if t_final is None else [t_final]

        # Build Drake trajectories for the goals (zero-order hold) and the jack
        # poses (cubic with continuous second derivatives).
        # The zero-order hold requires a final break time but will ignore the
        # final pose.  Add the final time and repeat the last goal pose to
        # satisfy.
        times_of_goals = np.concatenate((
            self.times_of_new_goals, [self.times[-1]]))
        goal_poses = np.vstack((self.goal_poses, self.goal_poses[-1, :]))
        goal_traj = PiecewisePolynomial.ZeroOrderHold(
            breaks=times_of_goals, samples=goal_poses.T)

        jack_quats, jack_xyzs = self.jack_poses[:, :4], self.jack_poses[:, 4:7]
        jack_quaternions = [Quaternion(q/np.linalg.norm(q)) for q in jack_quats]
        position_trajectory = \
            PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                breaks=self.times,
                samples=jack_xyzs.T,
                sample_dot_at_start=np.zeros(3),
                sample_dot_at_end=np.zeros(3)
            )
        orientation_trajectory = PiecewiseQuaternionSlerp(
            breaks=self.times,
            quaternions=jack_quaternions
        )
        # Sadly the more fool-proof PiecewisePose outputs 4x4 homogeneous
        # transform matrices, but TrajectorySource needs a column vector.  Use
        # StackedTrajectory instead, and use caution when interpreting the
        # output ordering.
        jack_quat_pos_traj = StackedTrajectory()
        jack_quat_pos_traj.Append(orientation_trajectory)
        jack_quat_pos_traj.Append(position_trajectory)

        self.vid = ProgressVideoMaker(
            save_dir=self.save_folder, open_meshcat=True, per_goal=per_goal,
            is_push_T=log_is_push_t(self.log_filepaths[0]))
        for i, (t0, tf) in enumerate(zip(t_inits, t_finals)):
            self.vid.visualize(goal_traj, jack_quat_pos_traj,
                               t_init=t0, t_final=tf, goal=i+1)

    # TODO
    def generate_error_plot_video(self):
        pass

    def generate_demo_video(self):
        self._extract_information()
        self._get_trajectory_tolerances()

        # Generate plot with mode shading and switch reasons.
        inspect_mode_switching_by_goal(
            times=self.times,
            is_c3_mode_flags=self.is_c3_mode_flags,
            switch_reasons=self.switch_reasons,
            pos_errors=self.pos_errors,
            rad_errors=self.rad_errors,
            pos_tol=self.pos_tol,
            rad_tol=self.rad_tol,
            goal_num=0,
            log_folder=self.save_folder
        )

        breakpoint()

    def print_statistics(self):
        self._extract_information()
        self._compute_times_to_thresholds()

        pos_thresholds = POS_SUCCESS_THRESHOLDS[
            POS_SUCCESS_THRESHOLDS >= self.pos_tol]
        rad_thresholds = RAD_SUCCESS_THRESHOLDS[
            RAD_SUCCESS_THRESHOLDS >= self.rad_tol]

        print(f'Number of trials: {self.times_to_thresholds.shape[0]}')

        for thresh_i, (pos_thresh, rad_thresh) in enumerate(
            zip(pos_thresholds, rad_thresholds)):
            print(f'\n=== Thresholds {pos_thresh}m {rad_thresh}rad ===')
            data = self.times_to_thresholds[:, thresh_i]
            print(f'Mean time to goal:  {np.mean(data)}')
            print(f'Standard deviation: {np.std(data)}')
            print(f'Median time to goal:  {np.median(data)}')
            print(f'Min time to goal:  {np.min(data)}')
            print(f'Max time to goal:  {np.max(data)}')


class ProgressVideoMaker:
    """Generates videos of the goal and jack poses over time, from the
    perspective of a jack-locked camera view and from a goal-locked camera view.
    """
    def __init__(self, save_dir: str, open_meshcat: bool = False,
                 per_goal: bool = False, is_push_T: bool = False):
        self.save_dir = save_dir if save_dir is not None else \
            file_utils.tmp_dir()
        self.open_meshcat = open_meshcat
        self.per_goal = per_goal
        self.is_push_T = is_push_T

    def visualize(self, goal_traj: PiecewisePolynomial,
                  jack_traj: PiecewisePolynomial, t_init: float,
                  t_final: float, goal: int = None):
        if self.per_goal:
            assert goal is not None, f'Goal must be specified if ' + \
                f'per_goal=True.'
            filename_end = f'goal_{goal}'
        else:
            filename_end = f'times_{int(t_init)}_{int(t_final)}'

        # Configure some things for the jack versus push T.
        goal_urdf = GOAL_T_URDF_PATH if self.is_push_T else GOAL_JACK_URDF_PATH
        actual_urdf = ACTUAL_T_URDF_PATH if self.is_push_T else \
            ACTUAL_JACK_URDF_PATH
        obj_locked_camera_quat = T_LOCKED_CAMERA_QUAT if self.is_push_T \
            else JACK_LOCKED_CAMERA_QUAT
        obj_locked_camera_offset = T_LOCKED_CAMERA_OFFSET if self.is_push_T \
            else JACK_LOCKED_CAMERA_OFFSET

        # Start building the Drake diagram.
        builder = DiagramBuilder()

        # Add the jack and goal triad to the plant.
        mbp_config = MultibodyPlantConfig(time_step=0)
        plant, scene_graph = AddMultibodyPlant(
            mbp_config, builder)
        parser = Parser(plant)
        goal_vis = parser.AddModels(goal_urdf)[0]
        jack_vis = parser.AddModels(actual_urdf)[0]
        goal_camera_vis = parser.AddModels(CAMERA_URDF_PATH)[0]
        jack_camera_vis = parser.AddModels(SECOND_CAMERA_URDF_PATH)[0]
        plant.RegisterVisualGeometry(
            plant.world_body(), RigidTransform(p=np.array([0, 0, -0.029])),
            HalfSpace(), 'table', np.array([0.5, 0.5, 0.5, 0.5]))
        plant.Finalize()
        plant.set_name('plant')

        # Add a meshcat visualizer.
        if self.open_meshcat:
            if hasattr(self, 'meshcat'):
                self.meshcat.Delete()
            else:
                self.meshcat = StartMeshcat()
            MeshcatVisualizer.AddToBuilder(
                builder, scene_graph, self.meshcat)

        # Add a vtk renderer; necessary to add video writers not with the
        # VideoWriter.AddToBuilder method.
        if not scene_graph.HasRenderer('vtk'):
            scene_graph.AddRenderer('vtk', MakeRenderEngineVtk(
                RenderEngineVtkParams()))

        # Add a goal-locked video writer (with fixed orientation and translation
        # offset).
        g_intrinsics = CameraInfo(
            width=VIDEO_PIXELS[1], height=VIDEO_PIXELS[0], fov_y=CAM_FOV)
        g_clip = ClippingRange(0.01, 10.0)
        g_camera = DepthRenderCamera(
            RenderCameraCore("vtk", g_intrinsics, g_clip, RigidTransform()),
            DepthRange(0.01, 10.0)
        )
        g_sensor = RgbdSensor(
            plant.GetBodyFrameIdOrThrow(
                plant.GetBodyByName('invisible_body').index()),
            RigidTransform(),
            g_camera
        )
        builder.AddSystem(g_sensor)
        builder.Connect(scene_graph.GetOutputPort('query'),
                        g_sensor.GetInputPort('geometry_query'))
        video_writer_goal = VideoWriter(
            filename=op.join(
                self.save_dir, f'goal_{filename_end}.mp4'),
            fps=VIDEO_FPS,
            backend="cv2"
        )
        builder.AddSystem(video_writer_goal)
        video_writer_goal.ConnectRgbdSensor(builder=builder, sensor=g_sensor)

        # Add a jack-locked video writer (with fixed orientation and translation
        # offset).
        j_intrinsics = CameraInfo(
            width=VIDEO_PIXELS[1], height=VIDEO_PIXELS[0], fov_y=CAM_FOV)
        j_clip = ClippingRange(0.01, 10.0)
        j_camera = DepthRenderCamera(
            RenderCameraCore("vtk", j_intrinsics, j_clip, RigidTransform()),
            DepthRange(0.01, 10.0)
        )
        j_sensor = RgbdSensor(
            plant.GetBodyFrameIdOrThrow(
                plant.GetBodyByName('second_invisible_body').index()),
            RigidTransform(),
            j_camera
        )
        builder.AddSystem(j_sensor)
        builder.Connect(scene_graph.GetOutputPort('query'),
                        j_sensor.GetInputPort('geometry_query'))
        video_writer_jack = VideoWriter(
            filename=op.join(
                self.save_dir, f'current_{filename_end}.mp4'),
            fps=VIDEO_FPS,
            backend="cv2"
        )
        builder.AddSystem(video_writer_jack)
        video_writer_jack.ConnectRgbdSensor(builder=builder, sensor=j_sensor)

        # Build the diagram.
        diagram = builder.Build()
        simulator = Simulator(diagram)
        context = simulator.get_context()
        diagram.ForcedPublish(context)

        for t in tqdm(np.arange(t_init, t_final, 1.0/VIDEO_FPS)):
            # Update the visualization.
            goal_camera_xyz = goal_traj.value(t)[4:7] + obj_locked_camera_offset
            jack_camera_xyz = jack_traj.value(t)[4:7] + obj_locked_camera_offset
            configs = np.vstack((
                goal_traj.value(t),
                jack_traj.value(t),
                obj_locked_camera_quat, goal_camera_xyz,
                obj_locked_camera_quat, jack_camera_xyz
            ))
            context = simulator.get_context()
            plant_context = plant.GetMyMutableContextFromRoot(context)
            plant.SetPositions(plant_context, configs)
            diagram.ForcedPublish(context)

            vw_context_front = video_writer_goal.GetMyContextFromRoot(context)
            video_writer_goal._publish(vw_context_front)
            vw_context_jack = video_writer_jack.GetMyContextFromRoot(
                context)
            video_writer_jack._publish(vw_context_jack)

        video_writer_goal.Save()
        video_writer_jack.Save()


class DemoVideoMaker:
    """Generates videos of the Franka manipulating the jack."""
    def __init__(self, save_dir: str, open_meshcat: bool = False):
        self.save_dir = save_dir if save_dir is not None else \
            file_utils.tmp_dir()
        self.open_meshcat = open_meshcat

    def visualize(self, goal_traj: PiecewisePolynomial,
                  jack_traj: PiecewisePolynomial, t_init: float,
                  t_final: float):
        builder = DiagramBuilder()

        # Add the jack and goal triad to the plant.
        mbp_config = MultibodyPlantConfig(time_step=0)
        plant, scene_graph = AddMultibodyPlant(
            mbp_config, builder)
        parser = Parser(plant)
        goal_vis = parser.AddModels(GOAL_JACK_URDF_PATH)[0]
        jack_vis = parser.AddModels(ACTUAL_JACK_URDF_PATH)[0]
        goal_camera_vis = parser.AddModels(CAMERA_URDF_PATH)[0]
        jack_camera_vis = parser.AddModels(SECOND_CAMERA_URDF_PATH)[0]
        plant.RegisterVisualGeometry(
            plant.world_body(), RigidTransform(p=np.array([0, 0, -0.029])),
            HalfSpace(), 'table', np.array([0.5, 0.5, 0.5, 0.5]))
        plant.Finalize()
        plant.set_name('plant')

        # Add a meshcat visualizer.
        if self.open_meshcat:
            if hasattr(self, 'meshcat'):
                self.meshcat.Delete()
            else:
                self.meshcat = StartMeshcat()
            MeshcatVisualizer.AddToBuilder(
                builder, scene_graph, self.meshcat)

        # Add a vtk renderer; necessary to add video writers not with the
        # VideoWriter.AddToBuilder method.
        if not scene_graph.HasRenderer('vtk'):
            scene_graph.AddRenderer('vtk', MakeRenderEngineVtk(
                RenderEngineVtkParams()))

        # Add a goal-locked video writer (with fixed orientation and translation
        # offset).
        g_intrinsics = CameraInfo(
            width=VIDEO_PIXELS[1], height=VIDEO_PIXELS[0], fov_y=CAM_FOV)
        g_clip = ClippingRange(0.01, 10.0)
        g_camera = DepthRenderCamera(
            RenderCameraCore("vtk", g_intrinsics, g_clip, RigidTransform()),
            DepthRange(0.01, 10.0)
        )
        g_sensor = RgbdSensor(
            plant.GetBodyFrameIdOrThrow(
                plant.GetBodyByName('invisible_body').index()),
            RigidTransform(),
            g_camera
        )
        builder.AddSystem(g_sensor)
        builder.Connect(scene_graph.GetOutputPort('query'),
                        g_sensor.GetInputPort('geometry_query'))
        video_writer_goal = VideoWriter(
            filename=op.join(
                self.save_dir, f'goal_{int(t_init)}_{int(t_final)}.mp4'),
            fps=VIDEO_FPS,
            backend="cv2"
        )
        builder.AddSystem(video_writer_goal)
        video_writer_goal.ConnectRgbdSensor(builder=builder, sensor=g_sensor)

        # Build the diagram.
        diagram = builder.Build()
        simulator = Simulator(diagram)
        context = simulator.get_context()
        diagram.ForcedPublish(context)

        for t in tqdm(np.arange(t_init, t_final, 1.0/VIDEO_FPS)):
            # Update the visualization.
            goal_camera_xyz = goal_traj.value(t)[4:7] + JACK_LOCKED_CAMERA_OFFSET
            jack_camera_xyz = jack_traj.value(t)[4:7] + JACK_LOCKED_CAMERA_OFFSET
            configs = np.vstack((
                goal_traj.value(t),
                jack_traj.value(t),
                JACK_LOCKED_CAMERA_QUAT, goal_camera_xyz,
                JACK_LOCKED_CAMERA_QUAT, jack_camera_xyz
            ))
            context = simulator.get_context()
            plant_context = plant.GetMyMutableContextFromRoot(context)
            plant.SetPositions(plant_context, configs)
            diagram.ForcedPublish(context)

            vw_context_front = video_writer_goal.GetMyContextFromRoot(context)
            video_writer_goal._publish(vw_context_front)

        video_writer_goal.Save()


@click.group()
def cli():
    pass


@cli.command('mjpc')
@click.argument('pickle-dir', type=click.Path(exists=True), required=True)
@click.option('--interactive', is_flag=True,
              help='Run in interactive mode')
@click.option('--save-to', type=str, default=None,
              help='Save the results to a folder of provided name')
@click.option('--delete', is_flag=True,
              help='Delete the save folder if it exists')
def mjpc_command(pickle_dir: str, interactive: bool, save_to: str,
                 delete: bool):
    if save_to is not None:
        save_folder = op.join(file_utils.tmp_dir(), save_to)
        if op.exists(save_folder) and delete:
            shutil.rmtree(save_folder)
        elif not op.exists(save_folder):
            os.makedirs(save_folder)
    else:
        save_folder = None

    # Look for ours_sim.pickle and multiple MJPC files of this type:
    # mjpc_ee_vel_0-24.pickle
    result_analyzers_by_ee_vel = {}
    result_analyzer_ours = None

    # Iterate over all files in the directory
    for filename in os.listdir(pickle_dir):
        if filename.endswith('.pickle'):
            filepath = op.join(pickle_dir, filename)
            print(f'Loading {filepath}... ', end='')

            with open(filepath, 'rb') as f:
                ra = pickle.load(f)
            assert isinstance(ra, ResultsAnalyzer)
            ra.save_folder = save_folder

            if filename == 'ours_sim.pickle':
                result_analyzer_ours = ra
            elif filename.startswith('mjpc_ee_vel_'):
                ee_vel_str = filename.split('_')[3].split('.')[0]
                ones = int(ee_vel_str.split('-')[0])
                decimals = 0.01 * int(ee_vel_str.split('-')[1])
                ee_vel = ones + decimals
                result_analyzers_by_ee_vel[ee_vel] = ra

            print(f'Done.')

    if interactive:
        global global_is_interactive
        global_is_interactive = True
        plt.ion()

    for ee_vel, result_analyzer in result_analyzers_by_ee_vel.items():
        result_analyzer.visualize_goals_with_violations(
            title_suffix=f'EE Velocity cost = {ee_vel}')
    result_analyzer_ours.visualize_goals_with_violations(title_suffix=f'Ours')
    joint_mjpc_cdf(result_analyzers_by_ee_vel, result_analyzer_ours)


@cli.command('multi')
@click.argument('log-folders', type=click.Path(exists=True), nargs=-1,
                required=True)
@click.option('--save-to', type=str, default=None,
              help='Save the results to a folder of provided name')
@click.option('--interactive', is_flag=True,
              help='Run in interactive mode')
@click.option('--video', is_flag=True,
              help='Generate video')
@click.option('--demo', is_flag=True,
              help='Generate demonstration video with modes and samples')
@click.option('--trim-times', is_flag=True,
              help='Print ffmpeg trim times for video generation')
@click.option('--overlay', is_flag=True,
              help='Overlay the meshcat video on the phone videos')
@click.option('--delete', is_flag=True,
              help='Delete the save folder if it exists')
@click.option('--export-name', type=str,
              help='Export the ResultsAnalyzer object')

def multi_command(log_folders: Tuple[str], save_to: str, interactive: bool,
                  video: bool, demo: bool, trim_times: bool, overlay: bool,
                  delete: bool, export_name: str):
    # Turn the folders into filepaths.
    log_type = None
    log_filepaths = []
    for log_folder in log_folders:
        log_folder = log_folder[:-1] if log_folder[-1] == '/' else log_folder
        log_number = log_folder.split('/')[-1][:6]
        hwlog_filepath = op.join(log_folder, f'hwlog-{log_number}')
        simlog_filepath = op.join(log_folder, f'simlog-{log_number}')
        mjpclog_filepath = op.join(log_folder, f'mjpclog-{log_number}')
        if op.exists(hwlog_filepath):
            log_filepath = hwlog_filepath
            if log_type == None:
                log_type = 'hardware'
            elif log_type != 'hardware':
                raise RuntimeError(f'Can only combine logs of the same ' + \
                                   f'type:  got {log_type} and hardware.')
        elif op.exists(simlog_filepath):
            log_filepath = simlog_filepath
            if log_type == None:
                log_type = 'simulation'
            elif log_type != 'simulation':
                raise RuntimeError(f'Can only combine logs of the same ' + \
                                   f'type:  got {log_type} and simulation.')
        elif op.exists(mjpclog_filepath):
            log_filepath = mjpclog_filepath
            if log_type == None:
                log_type = 'mjpc'
            elif log_type != 'mjpc':
                raise RuntimeError(f'Can only combine logs of the same ' + \
                                   f'type:  got {log_type} and mjpc.')
        else:
            raise ValueError(f'Could not find simlog, hwlog, or mjpclog in:' + \
                             f' {log_folder}')
        print(f'Parsing {log_type} log at: {log_filepath}')
        log_filepaths.append(log_filepath)
    print('')

    channels_and_lcmt = MINIMAL_CHANNELS_AND_LCMT_FOR_MJPC \
        if log_type == 'mjpc' else MINIMAL_CHANNELS_AND_LCMT_FOR_VIDEO

    # Print trim times or make overlay videos of meshcat vis on top of camera.
    if (trim_times and not video) or overlay:
        for log_filepath in log_filepaths:
            print(f'\n#=== {log_filepath} ===#')
            results_analyzer = ResultsAnalyzer(
                log_filepaths=[log_filepath],
                channels=channels_and_lcmt.keys(),
                trim_bookends=True
            )
            if overlay:
                results_analyzer.overlay_videos(per_goal=trim_times)
            else:
                results_analyzer.print_trim_times()
        exit()

    # Set the folder to save to.
    if save_to is not None:
        save_folder = op.join(file_utils.tmp_dir(), save_to)
        if op.exists(save_folder) and delete:
            shutil.rmtree(save_folder)
        elif not op.exists(save_folder):
            os.makedirs(save_folder)
    else:
        save_folder = None

    if interactive:
        global global_is_interactive
        global_is_interactive = True
        plt.ion()

    results_analyzer = ResultsAnalyzer(
        log_filepaths=log_filepaths,
        channels=channels_and_lcmt.keys(),
        save_folder=save_folder,
        verbose=False,
        trim_bookends=not demo
    )

    if export_name is not None:
        print(f'Will export ResultsAnalyzer and exit.')
        export_name += '.pickle' if not export_name.endswith('.pickle') else ''
        export_filepath = op.join(EXPORT_FOLDER, export_name)
        results_analyzer.prepare_and_export(export_filepath)
        exit()

    if demo:
        if video:
            results_analyzer.generate_goal_video(t_init=6, t_final=30)
        results_analyzer.generate_demo_video()
        breakpoint()
        exit()

    if video:
        results_analyzer.generate_goal_video(per_goal=trim_times)
        exit()

    # Note:  detecting hardware violations takes a while to run since the franka
    # states are high-frequency.  Only uncomment the below if desired.
    # results_analyzer.visualize_goals_with_violations()

    # These plots are cheap to produce at this point, so generate them all.
    results_analyzer.visualize_cdf()
    results_analyzer.visualize_goal_success()
    results_analyzer.inspect_mode_switching_by_goal()
    results_analyzer.visualize_time_histograms()
    results_analyzer.visualize_time_to_goal_vs_error()
    results_analyzer.print_statistics()
    breakpoint()


@cli.command('single')
@click.argument('log-folder', type=click.Path(exists=True), required=True)
@click.option('--start', type=float, default=0.0,
              help='Start time into the log to begin parsing')
@click.option('--end', type=float, default=1e12,
              help='End time into the log to stop parsing')
@click.option('--buffer-vis', is_flag=True,
              help='Visualize the buffer of samples throughout time range')
@click.option('--inspect-switching', is_flag=True,
              help='Inspect switching between C3 and repositioning')
@click.option('--visualize-goal-completion', is_flag=True,
              help='Visualize the completion of goals')
@click.option('--lcm-traffic-debug', is_flag=True,
              help='Debug LCM traffic issues')
@click.option('--all', is_flag=True,
              help='Run all visualizations')
@click.option('--interactive', is_flag=True,
              help='Run in interactive mode')
@click.option('--present', is_flag=True,
              help='Generate presentable plots')
@click.option('--video', is_flag=True,
              help='Generate video')

def single_command(log_folder: str, start: float, end: float, buffer_vis: bool,
                   inspect_switching: bool, visualize_goal_completion: bool,
                   lcm_traffic_debug: bool, all: bool, interactive: bool,
                   present: bool, video: bool):
    # Turn the folder into a file path.
    log_folder = log_folder[:-1] if log_folder[-1] == '/' else log_folder
    log_number = log_folder.split('/')[-1][:6]
    log_filepath = op.join(log_folder, f'simlog-{log_number}')
    log_type = 'simulation'
    if not op.exists(log_filepath):
        log_filepath = op.join(log_folder, f'hwlog-{log_number}')
        log_type = 'hardware'
    if not op.exists(log_filepath):
        raise ValueError(f'Could not find simlog or hwlog in: {log_folder}')
    print(f'Parsing {log_type} log at: {log_filepath}\n')

    channels_and_lcmt = ALL_CHANNELS_AND_LCMT if lcm_traffic_debug \
        else MINIMAL_CHANNELS_AND_LCMT_FOR_VIDEO if present and video \
        else MINIMAL_CHANNELS_AND_LCMT if present \
        else CHANNELS_AND_LCMT_TO_SYNC

    if interactive:
        global global_is_interactive
        global_is_interactive = True
        plt.ion()

    results_analyzer = ResultsAnalyzer(
        log_filepaths=[log_filepath],
        channels=channels_and_lcmt.keys(),
        start_times=[start], end_times=[end],
        verbose=True
    )

    if video:
        results_analyzer.generate_goal_video()

    if lcm_traffic_debug:
        inspect_lcm_traffic(results_analyzer.messages_by_channel)
        exit()

    if visualize_goal_completion or all:
        results_analyzer.visualize_goal_success()

    if inspect_switching or all:
        results_analyzer.inspect_mode_switching_by_goal()

    if all:
        results_analyzer.visualize_cdf()
        results_analyzer.visualize_time_histograms()
        results_analyzer.visualize_time_to_goal_vs_error()

    # Visualize the buffer of samples.
    if buffer_vis or all:
        visualize_sample_buffer(results_analyzer.messages_by_channel)

    if interactive:
        breakpoint()


if __name__ == '__main__':
    cli()

