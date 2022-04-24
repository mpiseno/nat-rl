import os
import argparse
import copy
from threading import current_thread
from pathlib import Path

import gym
import numpy as np
import habitat

from stanford_habitat.measures import * # register

from nat_rl.utils.env_utils import *
from nat_rl.utils.imitation import make_rollouts_from_offline_data
from nat_rl.utils.args import get_args
from nat_rl.utils.common import count_trainable_params
from nat_rl.utils.evaluation import evaluate_success_rate, EvalCallback
from nat_rl.algos.bc import BC_
from nat_rl.models import CustomCombinedExtractor, CustomMultiInputActorCriticPolicy, CLIPExtractor

from imitation.algorithms import bc
from imitation.algorithms.bc import reconstruct_policy
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper


env_fns = {
    'pick_single_object-v0': make_habitat_pick_single_object_env,
    'gc_pick_single_object-v0': make_habitat_GC_pick_single_object_env,
    'gc_pick_fruit': make_GC_pick_fruit_env
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
    if args.make_video:
        video_dir = f'visuals/{args.env}/eval_{args.eval}'
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    success_rate = evaluate_success_rate(
        env, policy, num_eps=100,
        video_dir=video_dir,
        train_mode=False
    )

    print(f'Success rate: {success_rate}')


def get_policy(env, args):
    features_extractor_class = feature_extractors[args.feature_extractor]
    policy = CustomMultiInputActorCriticPolicy(
        observation_space=copy.deepcopy(env.observation_space),
        action_space=copy.deepcopy(env.action_space),
        lr_schedule=lambda x: args.lr, # This is not actually used in the BC algorithm, but its required by the policy class.
        features_extractor_class=features_extractor_class
    )

    num_params = count_trainable_params(policy)
    print(f'Number of trainable parameters: {num_params}')

    return policy


def train_IL(env, args):
    print('Generating transitions dataset from offline data')
    expert_traj_dir = env.env._config.DATASET.EXPERT_TRAJ_DIR
    rollouts = make_rollouts_from_offline_data(
        expert_traj_dir=expert_traj_dir, env=env
    )
    transitions = rollout.flatten_trajectories(trajectories=rollouts)
    print(f'Successfully generated transition data')

    policy = get_policy(env, args)

    bc_trainer = BC_(
        observation_space=copy.deepcopy(env.observation_space),
        action_space=copy.deepcopy(env.action_space),
        demonstrations=transitions,
        policy=policy,
        batch_size=args.batch_size,
        optimizer_kwargs = {
            'lr': args.lr
        },
        device=args.device
        #l2_weight=args.l2_weight # try regularization later
    )
    env.close() # Habitat only supports 1 environment per process

    eval_env_fn_kwargs = {'test_dataset': True}
    eval_env = env_fns[args.env](**eval_env_fn_kwargs)
    eval_callback = EvalCallback(eval_env, policy, num_eps=20)
    bc_trainer.train(
        n_epochs=args.n_IL_epochs,
        on_epoch_end=eval_callback,
    )

    save_model_path = os.path.join(
        args.logdir, f'bc/saved_models/{args.env}-{args.feature_extractor}-n_epochs={args.n_IL_epochs}.pt'
    )
    bc_trainer.save_policy(save_model_path)


def main():
    os.environ['HABITAT_SIM_LOG'] = 'quiet'
    args = get_args()
    assert args.env in env_fns, "Please choose a valid environment name"
    assert args.feature_extractor in feature_extractors, "Invalid feature extractor class for the policy"

    if args.eval is not None:
        env_fn_kwargs = {'test_dataset': args.eval == 'test'}
        env = env_fns[args.env](**env_fn_kwargs)
        eval_IL(env, args)
    else:
        env_fn_kwargs = {'test_dataset': False}
        env = env_fns[args.env](**env_fn_kwargs)
        
        train_IL(env, args)


if __name__ == '__main__':
    main()