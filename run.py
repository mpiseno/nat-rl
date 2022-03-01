import os
import argparse

import gym
import habitat

from nat_rl.envs import PickPlaceTableEnv
from nat_rl.trainers import SACTrainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--num-timesteps', type=int, default=1000)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    config_file = os.path.join(os.getcwd(), args.config)
    config = habitat.get_config(config_file)

    trainer = SACTrainer(env_class=PickPlaceTableEnv, env_kwargs={'config': config})
    if args.mode == 'train':
        trainer.learn(timesteps=args.num_timesteps)
    elif args.mode == 'eval':
        trainer.eval()


if __name__ == '__main__':
    main()
