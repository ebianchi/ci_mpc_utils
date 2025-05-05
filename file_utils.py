"""File utilities for interfacing with dairlib repo.  Requires ci_mpc_utils repo
to be installed at same directory as dairlib repo."""

import numpy as np
import os.path as op
import yaml


HEAD_DIR = op.abspath(op.dirname(op.dirname(__file__)))
DAIRLIB_DIR = op.join(HEAD_DIR, 'dairlib')

assert op.isdir(DAIRLIB_DIR), f'Did not find dairlib at {DAIRLIB_DIR}'


### Directories ###
def dairlib_dir():
    return DAIRLIB_DIR

def example_dir():
    return op.join(dairlib_dir(), 'examples/sampling_c3')

def subexample_dir(system: str):
    assert system is not None, f'Need to provide sub-example for sampling-' + \
        f'based C3.'
    sub_dir = op.join(example_dir(), system)
    assert op.exists(sub_dir), f'Did not find {sub_dir}'
    return sub_dir

### Filepaths ###
def urdf_dir():
    return op.join(example_dir(), 'urdf')

def jack_urdf_path():
    return op.join(urdf_dir(), 'jack.sdf')

def ground_urdf_path():
    return op.join(urdf_dir(), 'ground.urdf')

def end_effector_urdf_path():
    return op.join(urdf_dir(), 'end_effector_full.urdf')

### Load parameters from yaml files ###
def load_franka_sim_params(system: str):
    sub_dir = subexample_dir(system)
    yaml_file = op.join(sub_dir, 'parameters', 'franka_sim_params.yaml')
    with open(yaml_file) as f:
        sim_params = yaml.load(f, Loader=yaml.FullLoader)

    return sim_params

def load_p_world_to_franka(system: str):
    sim_params = load_franka_sim_params(system)
    return np.array(sim_params['p_world_to_franka'])

def load_p_franka_to_ground(system: str):
    sim_params = load_franka_sim_params(system)
    return np.array(sim_params['p_franka_to_ground'])

