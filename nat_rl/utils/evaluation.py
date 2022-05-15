import os
import time

import numpy as np
import torch

from habitat_sim.utils import viz_utils as vut
from stable_baselines3.common.policies import ActorCriticPolicy


SUCCESS_RATES_FNAME = 'success_rates.txt'


def run_single_episode(
    env,
    policy,
    goal_type,
    video_dir=None,
    train_mode=None
):
    obs = env.reset()
    success = False
    episode_id = env.habitat_env.current_episode.episode_id
    traj = [obs]
    while True:
        action, _ = policy.predict(obs, goal_type=goal_type, deterministic=True, train_mode=train_mode)
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


def evaluate_success_rate(env, policy, goal_type, num_eps=10, video_dir=None, train_mode=None):
    num_success = 0
    for _ in range(num_eps):
        success = run_single_episode(
            env, policy,
            goal_type=goal_type,
            video_dir=video_dir, train_mode=train_mode
        )

        if success:
            num_success += 1
    
    success_rate = num_success / num_eps
    return success_rate


def multiprocess_eval_target(env_fn, env_fn_kwargs, saved_policy_path):
    env = env_fn(**env_fn_kwargs)
    policy = torch.load(saved_policy_path, map_location='cuda')
    assert isinstance(policy, ActorCriticPolicy)

    num_eps = len(env.habitat_env.episodes)
    start = time.time()
    success_rate = evaluate_success_rate(
        env, policy, goal_type=env_fn_kwargs['goal_type'],
        num_eps=num_eps, train_mode=False
    )

    print(f'Evaluated {saved_policy_path} in {round(time.time() - start, 2)} seconds | success rate: {success_rate}')
    
    if 'final' in saved_policy_path:
        epoch = 100
    else:
        epoch = int(saved_policy_path.split('=')[-1][:-len('.pt')])
    
    env.close()
    return (epoch, success_rate)
    

def evaluate_grid(env_fn, env_fn_kwargs, saved_policies_dir, mp_ctx):
    policy_fnames = filter(
        lambda x: x.endswith('.pt'),
        os.listdir(saved_policies_dir)
    )
    multiprocess_args = []
    for policy_fname in policy_fnames:
        multiprocess_args.append(
            (
                env_fn,
                env_fn_kwargs,
                os.path.join(saved_policies_dir, policy_fname)
            )
        )
    
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    print(f'Using {num_cpus} CPUs')
    with mp_ctx.Pool(num_cpus) as pool:
       results = pool.starmap(multiprocess_eval_target, multiprocess_args)
        
    return results

    
    # for policy_file in policy_fnames:
    #     if 'final' in policy_file:
    #         cur_epoch = 100
    #     else:
    #         cur_epoch = int(policy_file.split('=')[-1][:-len('.pt')])

    #     # if cur_epoch < 20:
    #     #     continue

    #     policy = torch.load(
    #         os.path.join(saved_policies_dir, policy_file), map_location='cuda'
    #     )
    #     assert isinstance(policy, ActorCriticPolicy)

    #     num_eps = len(env.habitat_env.episodes)
    #     start = time.time()
    #     success_rate = evaluate_success_rate(
    #         env, policy,
    #         num_eps=num_eps, train_mode=False
    #     )
    #     print(f'Evaluated {os.path.join(saved_policies_dir, policy_file)} in {round(time.time() - start, 2)} seconds')
    #     log_fname = os.path.join(saved_policies_dir, SUCCESS_RATES_FNAME)
    #     if os.path.isfile(log_fname):
    #         print(f'removing old log file')
    #         os.remove(log_fname)
    #     with open(log_fname, 'a') as f:
    #         f.write(f'{cur_epoch},{round(success_rate, 3)}\n')        


class EvalCallback:
    def __init__(self, eval_env=None, policy=None, num_eps=10, call_freq=10, save_dir='logs'):
        self.eval_env = eval_env
        self.policy = policy
        self.num_eps = num_eps
        self.save_dir = save_dir
        self.call_freq = call_freq
        self.n_calls = 0
    
    def __call__(self):
        if self.n_calls % self.call_freq == 0:
            if self.eval_env is not None:
                success_rate = evaluate_success_rate(
                    self.eval_env,
                    self.policy,
                    num_eps=self.num_eps,
                    train_mode=True
                )
                print(f'SuccessRateMetric | epoch: {self.n_calls} | success rate: {success_rate}')

            policy_path = os.path.join(self.save_dir, f'epoch={self.n_calls}.pt')
            torch.save(self.policy, policy_path)

        self.n_calls += 1

