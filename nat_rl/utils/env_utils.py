'''
Functions for instantiating specific habitat environments based on a config YAML file.
'''

import os

import habitat

from stanford_habitat.envs import RearrangeRLEnv, GCRearrangeRLEnv, HabitatArmActionWrapper
from stanford_habitat.measures import * # register
from stanford_habitat.tasks import (
    SimplePickTask, SimplePickPlaceTask,
    SpacialReasoningTask
) 
from stanford_habitat.datasets.rearrange_datasets import RearrangeDatasetV1


PICK_SINGLE_OBJECT_CONFIG = "configs/pick_task/pick_single_object.yaml"
GC_PICK_SINGLE_OBJECT_CONFIG = "configs/pick_task/pick_single_object_GC.yaml"
PICK_FRUIT_CONFIG = "configs/pick_task/pick_fruit/pick_fruit.yaml"
SPACIAL_REASONING_CONFIG = "configs/pickplace_tasks/spatial_reasoning_datagen/spatial_reasoning.yaml"

DEFAULT_ENV_OPTIONS = {'test_dataset': False}


def insert_test_dataset(config):
    '''Modifies the config file to use a test dataset'''
    config.defrost()
    path = config.DATASET.DATA_PATH.split('/')
    dataset_fname = path[-1][:-len('.json.gz')]
    path[-1] = dataset_fname + '_test.json.gz' # assume there is a test dataset called this
    config.DATASET.DATA_PATH = '/'.join(path)

    expert_traj_path = config.DATASET.EXPERT_TRAJ_DIR.split('/')
    if expert_traj_path[-1] == '':
        _ = expert_traj_path.pop()

    expert_traj_path[-1] = expert_traj_path[-1] + '_test/'
    config.DATASET.EXPERT_TRAJ_DIR = '/'.join(expert_traj_path)
    config.freeze()
    return config


def make_habitat_pick_single_object_env(env_options=DEFAULT_ENV_OPTIONS):
    # TODO: replace hard coded env path with something from env_config
    config_path = os.path.join(os.getcwd(), PICK_SINGLE_OBJECT_CONFIG)
    config = habitat.get_config(config_path)
    if env_options['test_dataset']:
        config = insert_test_dataset(config)

    env = RearrangeRLEnv(config=config)
    env = HabitatArmActionWrapper(env)

    return env


def make_habitat_GC_pick_single_object_env():
    # TODO: replace hard coded env path with something from env_config
    config_path = os.path.join(os.getcwd(), GC_PICK_SINGLE_OBJECT_CONFIG)
    config = habitat.get_config(config_path)
    env = GCRearrangeRLEnv(config=config)
    env = HabitatArmActionWrapper(env)

    return env


def make_pick_fruit_env():
    config_path = os.path.join(os.getcwd(), PICK_FRUIT_CONFIG)
    config = habitat.get_config(config_path)
    env = RearrangeRLEnv(config=config)
    env = HabitatArmActionWrapper(env)
    return env


def make_GC_pick_fruit_env(test_dataset=False, env_kwargs={}):
    config_path = os.path.join(os.getcwd(), PICK_FRUIT_CONFIG)
    config = habitat.get_config(config_path)
    if test_dataset == True:
        config = insert_test_dataset(config)

    env = GCRearrangeRLEnv(config=config, **env_kwargs)
    env = HabitatArmActionWrapper(env)
    return env


def make_GC_spacial_reasoning_env(test_dataset=False, env_kwargs={}):
    config_path = os.path.join(os.getcwd(), SPACIAL_REASONING_CONFIG)
    config = habitat.get_config(config_path)
    if test_dataset == True:
        config = insert_test_dataset(config)

    env = habitat.Env(config=config)
    env = HabitatArmActionWrapper(env)
    return env