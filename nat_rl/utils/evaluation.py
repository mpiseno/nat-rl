import os

import numpy as np
import torch

from habitat_sim.utils import viz_utils as vut


def run_single_episode(
    env,
    policy,
    video_dir=None,
    train_mode=None
):
    obs = env.reset()
    success = False
    episode_id = env.habitat_env.current_episode.episode_id
    traj = [obs]
    while True:
        action, _ = policy.predict(obs, deterministic=True, train_mode=train_mode)
        obs, rew, done, info = env.step(action)
        traj.append(obs)

        if done:
            if env.episode_success():
                success = True

            break
    
    make_video = video_dir is not None
    if make_video:
        video_file_path = os.path.join(video_dir, f'episodeid={episode_id}.mp4')
        video_writer = vut.get_fast_video_writer(video_file_path, fps=30)
        for obs in traj:
            video_writer.append_data(obs['robot_third_rgb'])

        video_writer.close()
    
    return success


def evaluate_success_rate(env, policy, num_eps=10, video_dir=None, train_mode=None):
    num_success = 0
    for _ in range(num_eps):
        success = run_single_episode(
            env, policy, 
            video_dir=video_dir, train_mode=train_mode
        )

        if success:
            num_success += 1
    
    success_rate = num_success / num_eps
    return success_rate



class EvalCallback:
    def __init__(self, eval_env, policy, num_eps=10):
        self.eval_env = eval_env
        self.policy = policy
        self.num_eps = num_eps
        self.n_calls = 0
    
    def __call__(self):
        success_rate = evaluate_success_rate(
            self.eval_env,
            self.policy,
            num_eps=self.num_eps,
            train_mode=True
        )

        self.n_calls += 1
        print(f'SuccessRateMetric | epoch: {self.n_calls} | success rate: {success_rate}')

