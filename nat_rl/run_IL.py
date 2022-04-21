import os
import argparse
from threading import current_thread

import gym
import numpy as np
import habitat

from stanford_habitat.measures import * # register

from nat_rl.utils.env_utils import make_habitat_pick_single_object_env, make_habitat_GC_pick_single_object_env
from nat_rl.utils.imitation import make_rollouts_from_offline_data
from nat_rl.utils.args import get_args
from nat_rl.utils.common import count_trainable_params, evaluate_success_rate
from nat_rl.algos.bc import BC_
from nat_rl.models import CustomCombinedExtractor, CustomMultiInputActorCriticPolicy, CNNFeatureExtractor

from imitation.algorithms import bc
from imitation.algorithms.bc import reconstruct_policy
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper


EXPERT_TRAJ_DIR = 'data/expert_trajs/'
env_fns = {
    'pick_single_object-v0': make_habitat_pick_single_object_env,
    'gc_pick_single_object-v0': make_habitat_GC_pick_single_object_env
}


def eval_IL(env, args, eval_train=True):
    policy = reconstruct_policy('trained_policy.pt')
    success_rate = evaluate_success_rate(env, policy, num_eps=50, make_video=True, eval_train=eval_train)
    print(f'Success rate: {success_rate}')


def main():
    os.environ['HABITAT_SIM_LOG'] = 'quiet'
    args = get_args()
    assert args.env in env_fns, "Please choose a valid environment name"

    env_config = {'eval_dataset': True}
    env = env_fns[args.env](env_config)

    if args.eval:
        eval_train = not env_config['eval_dataset']
        eval_IL(env, args, eval_train=eval_train)
    else:
        print('Generating transitions dataset from offline data')
        rollouts = make_rollouts_from_offline_data(
            expert_traj_dir=EXPERT_TRAJ_DIR, env=env
        )
        transitions = rollout.flatten_trajectories(trajectories=rollouts)
        print(f'Successfully generated transition data')

        policy = CustomMultiInputActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda x: args.lr,
            features_extractor_class=CustomCombinedExtractor
        )
        num_params = count_trainable_params(policy)
        print(f'Number of trainable parameters: {num_params}')

        bc_trainer = BC_(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            policy=policy,
            batch_size=args.batch_size,
            optimizer_kwargs = {
                'lr': args.lr
            }
            #l2_weight=args.l2_weight # try regularization later
        )
        print(bc_trainer._bc_logger._logger.dir)
        bc_trainer.train(
            n_epochs=args.n_IL_epochs,
            reset_tensorboard=True
        )
        bc_trainer.save_policy('trained_policy.pt')


if __name__ == '__main__':
    main()