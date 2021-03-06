'''
Functions for instantiating specific habitat environments based on a config YAML file.
'''

import os

import habitat

from stanford_habitat.envs import ImageGCRLEnv, CLIPGCRLEnv, CLIPSearch_GCRLEnv, HabitatArmActionWrapper
from stanford_habitat.measures import * # register
from stanford_habitat.tasks import (
    SimplePickTask, SimplePickPlaceTask,
    SpacialReasoningTask
) 
from stanford_habitat.datasets.rearrange_datasets import RearrangeDatasetV1


PICK_SINGLE_OBJECT_CONFIG = "configs/pick_task/pick_single_object.yaml"
GC_PICK_SINGLE_OBJECT_CONFIG = "configs/pick_task/pick_single_object_GC.yaml"
PICK_FRUIT_CONFIG = "configs/pick_task/pick_fruit/pick_fruit.yaml"
SPATIAL_REASONING_CONFIG = "configs/pickplace_tasks/spatial_reasoning/spatial_reasoning.yaml"

DEFAULT_ENV_OPTIONS = {
    'test_dataset': False,
    'goal_type': None,
    'env_kwargs': {}
}


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


def make_GC_env(
    config_path, 
    test_dataset,
    goal_type,
    env_kwargs
):
    config = habitat.get_config(config_path)
    if test_dataset == True:
        config = insert_test_dataset(config)

    if goal_type == 'image':
        env = ImageGCRLEnv(config=config, **env_kwargs)
    elif goal_type in ['clip_img', 'clip_lang']:
        env = CLIPGCRLEnv(config=config, goal_type=goal_type)
    elif goal_type in ['clip_lang_plus_init', 'clip_lang_plus_init_nn']:
        env = CLIPSearch_GCRLEnv(config=config, goal_type=goal_type)

    env = HabitatArmActionWrapper(env)
    return env


def make_GC_pick_fruit_env(test_dataset=False, goal_type='clip', env_kwargs={}):
    config_path = os.path.join(os.getcwd(), PICK_FRUIT_CONFIG)
    return make_GC_env(config_path, test_dataset, goal_type, env_kwargs)


def make_GC_spacial_reasoning_env(test_dataset=False, goal_type='image', env_kwargs={}):
    config_path = os.path.join(os.getcwd(), SPATIAL_REASONING_CONFIG)
    return make_GC_env(config_path, test_dataset, goal_type, env_kwargs)