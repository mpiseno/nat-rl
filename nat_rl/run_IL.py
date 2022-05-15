import os
import argparse
import copy
import time
from threading import current_thread
from pathlib import Path

import gym
import numpy as np
import habitat

from gym import spaces

#from stanford_habitat.measures import * # register

from nat_rl.utils.env_utils import *
from nat_rl.utils.imitation import generate_offline_trajectories, generate_dummy_trajectories
from nat_rl.utils.args import get_args
from nat_rl.utils.common import count_trainable_params, get_policy, seed_all
from nat_rl.utils.evaluation import evaluate_success_rate, EvalCallback
from nat_rl.algos.bc import BC_
from nat_rl.models import CustomCombinedExtractor, CustomMultiInputActorCriticPolicy, CLIPExtractor

from imitation.algorithms import bc
from imitation.algorithms.bc import reconstruct_policy
from imitation.data import rollout


env_fns = {
    'gc_pick_single_object-v0': make_habitat_GC_pick_single_object_env,
    'gc_pick_fruit': make_GC_pick_fruit_env,
    'gc_spatial_reasoning': make_GC_spacial_reasoning_env
}

feature_extractors = {
    'CNN': CustomCombinedExtractor,
    'CLIP': CLIPExtractor
}


def eval_IL(env, args):
    assert args.load_policy_path is not None, "Please specify a path to the saved policy (use --load_policy_path)"
    
    print(f'Evaluating {args.env} with {args.eval} dataset')

    policy = reconstruct_policy(args.load_policy_path)

    video_dir = None
    if args.make_video == True:
        video_dir = f'visuals/{args.env}_CLIP/eval_{args.eval}'
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    success_rate = evaluate_success_rate(
        env, policy, num_eps=100,
        video_dir=video_dir,
        train_mode=False
    )

    print(f'Success rate: {success_rate}')


def format_experiment_dir(args, seed):
    return os.path.join(
            args.logdir,
            f'bc/saved_models/{args.env}_{args.feature_extractor}_seed={seed}_ts-{int(time.time())}/'
    )


def train_IL(env, args, clip, seed):
    expert_traj_dir = env.env._config.DATASET.EXPERT_TRAJ_DIR

    print(f'Generating transitions dataset from {expert_traj_dir}')
    rollouts = generate_offline_trajectories(
        expert_traj_dir=expert_traj_dir, clip=clip, embed_goal_only=True
    )
    #rollouts = generate_dummy_trajectories(n_trajs=50)
    transitions = rollout.flatten_trajectories(trajectories=rollouts)
    print(f'Successfully generated transition data')

    policy = get_policy(env, args, feature_extractors)

    seed_all(seed)
    bc_trainer = BC_(
        observation_space=policy.observation_space,
        action_space=policy.action_space,
        demonstrations=transitions,
        policy=policy,
        batch_size=args.batch_size,
        l2_weight=args.l2_weight,
        optimizer_kwargs = {
            'lr': args.lr
        },
        device=args.device
    )
    env.close() # Don't need the environment anymore so close it to save memory

    print(f'''
        policy: {policy}\n
        device: {policy.device}
    ''')

    save_model_dir = format_experiment_dir(args, seed)
    Path(save_model_dir).mkdir(parents=True, exist_ok=True)
    eval_callback = EvalCallback(eval_env=None, policy=policy, call_freq=5, save_dir=save_model_dir)
    bc_trainer.train(
        n_epochs=args.n_IL_epochs,
        on_epoch_end=eval_callback,
    )
    bc_trainer.save_policy(os.path.join(save_model_dir, 'final.pt'))


def run_experiments(args):
    for seed in args.seeds:
        clip = args.feature_extractor == 'CLIP'
        env_fn_kwargs = {
            'test_dataset': False,
            'goal_type': 'clip_img' if clip else 'image',
            'env_kwargs': {
                'load_goals': False
            }
        }
        env = env_fns[args.env](**env_fn_kwargs)
        
        train_IL(env, args, clip, seed)


def main():
    os.environ['HABITAT_SIM_LOG'] = 'quiet'
    args = get_args()

    assert args.env in env_fns, "Please choose a valid environment name"
    assert args.feature_extractor in feature_extractors, "Invalid feature extractor class for the policy"

    if args.eval is not None:
        env_fn_kwargs = {
            'test_dataset': args.eval == 'test',
            'env_kwargs': {
                'goal_format': 'clip'
            }
        }
        env = env_fns[args.env](**env_fn_kwargs)
        eval_IL(env, args)
    else:
        run_experiments(args)


if __name__ == '__main__':
    main()