'''
Functions for instantiating specific habitat environments based on a config YAML file.
'''

import os

import habitat

from stanford_habitat.envs import RearrangeRLEnv, GCRearrangeRLEnv, HabitatArmActionWrapper
from stanford_habitat.measures import * # register
from stanford_habitat.tasks import SimplePickTask, SimplePickPlaceTask # register
from stanford_habitat.datasets.rearrange_datasets import RearrangeDatasetV1


PICK_SINGLE_OBJECT_CONFIG = "configs/pick_task/pick_single_object.yaml"
GC_PICK_SINGLE_OBJECT_CONFIG = "configs/pick_task/pick_single_object_GC.yaml"

DEFAULT_ENV_OPTIONS = {'eval_dataset': False}


def insert_eval_dataset(config):
    config.defrost()
    path = config.DATASET.DATA_PATH.split('/')
    dataset_fname = path[-1][:-8]
    path[-1] = dataset_fname + '_eval.json.gz'
    config.DATASET.DATA_PATH = '/'.join(path)
    config.freeze()
    return config


def make_habitat_pick_single_object_env(env_options=DEFAULT_ENV_OPTIONS):
    # TODO: replace hard coded env path with something from env_config
    config_path = os.path.join(os.getcwd(), PICK_SINGLE_OBJECT_CONFIG)
    config = habitat.get_config(config_path)
    if env_options['eval_dataset']:
        config = insert_eval_dataset(config)

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