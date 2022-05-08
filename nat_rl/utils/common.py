import os
import copy
import random
from pathlib import Path

import torch
import numpy as np

from gym import spaces

from habitat_sim.utils import viz_utils as vut
from nat_rl.models import CustomCombinedExtractor, CustomMultiInputActorCriticPolicy, CLIPExtractor


def count_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_count = sum([np.prod(p.size()) for p in model_parameters])
    return param_count


def expert_img_sort_fn(fname):
    key = ''.join([c if c.isdigit() else '' for c in fname])
    key = int(key) if key else -1
    return key


def get_policy(env, args, feature_extractors):
    features_extractor_class = feature_extractors[args.feature_extractor]
    print(f'Using feature extractor: {features_extractor_class}')
    
    if features_extractor_class == CLIPExtractor:
        obs_space = spaces.Dict({
            'robot_third_rgb': copy.deepcopy(env.observation_space.spaces['robot_third_rgb']),
            'goal': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(512,), # size of CLIP embeddings
                dtype=np.float32
            )
        })
    elif features_extractor_class == CustomCombinedExtractor:
        obs_space = copy.deepcopy(env.observation_space)

    policy = CustomMultiInputActorCriticPolicy(
        observation_space=obs_space,
        action_space=copy.deepcopy(env.action_space),
        net_arch=[dict(pi=[128, 64])],
        lr_schedule=lambda x: args.lr, # This is not actually used in the BC algorithm, but its required by the policy class.
        features_extractor_class=features_extractor_class
    )

    num_params = count_trainable_params(policy)
    print(f'Number of trainable parameters: {num_params}')

    return policy


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    