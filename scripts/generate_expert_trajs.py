import os
import argparse
from threading import current_thread

import gym
import numpy as np
import habitat

from PIL import Image
from habitat_sim.utils import viz_utils as hviz_utils
from habitat.utils.visualizations.utils import observations_to_image

from stanford_habitat.utils import make_traj, execute_traj
from stanford_habitat.measures import * # register
from nat_rl.utils.ik import get_ee_waypoints
from nat_rl.utils.env_utils import PICK_SINGLE_OBJECT_CONFIG, PICK_FRUIT_CONFIG, insert_test_dataset

'''
Uses IK to solve pick&place tasks and saves the trajectory buffer
'''

EXPERT_TRAJ_BASE_DIR = 'data/expert_trajs/'
env_configs = {
    'pick_single_object-v0': PICK_SINGLE_OBJECT_CONFIG,
    'pick_fruit': PICK_FRUIT_CONFIG,
    'pick_fruit_test': PICK_FRUIT_CONFIG
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='pick_single_object-v0')
    parser.add_argument('--make_video', action='store_true', default=False)
    parser.add_argument('--make_videos_from_data', action='store_true', default=False)
    args = parser.parse_args()
    return args


def make_IK_config(config):
    config.defrost()

    config.TASK.ACTIONS.ARM_ACTION.ARM_CONTROLLER = "ArmEEAction"
    config.TASK.SENSORS = [
        "RELATIVE_TARGET_POSITION_SENSOR",
        "JOINT_SENSOR",
        "IS_HOLDING_SENSOR",
        "END_EFFECTOR_SENSOR",
        "RELATIVE_RESTING_POS_SENSOR"
    ]
    
    config.freeze()
    return config


def make_IK_env(args):
    config_path = env_configs[args.env]
    config_path = os.path.join(os.getcwd(), config_path)

    env_config = habitat.get_config(config_path)
    env_config = make_IK_config(env_config)
    if '_test' in args.env:
        env_config = insert_test_dataset(env_config)

    env = habitat.Env(config=env_config)
    return env


def save_traj(traj, action_traj, episode_id, expert_traj_dir):
    traj_dir = os.path.join(expert_traj_dir, f'episodeid={episode_id}/')
    os.mkdir(traj_dir)
    for t, obs in enumerate(traj):
        obs_file = traj_dir + f'obs{t}.png'
        rgb_obs = obs['robot_third_rgb']
        im = Image.fromarray(rgb_obs)
        im.save(obs_file, 'png')
    
    action_fname = traj_dir + 'actions.npy'
    np.save(action_fname, action_traj.astype(np.float32))


def make_video(traj):
    pass


def generate_single_trajectory(env, args, ep_iter=-1, expert_traj_dir=None):
    video_writer = None
    ee_ctrl_lim = env._config.TASK.ACTIONS.ARM_ACTION.EE_CTRL_LIM
    obs = env.reset()
    cur_episode_id = env.current_episode.episode_id

    if ep_iter != -1:
        print(f'Generating episodeid={cur_episode_id} ({ep_iter + 1}/{len(env.episodes)})')
    else:
        print(f'Generating episodeid={cur_episode_id} | {len(env.episodes)}) left')

    traj = []
    traj.append(obs)

    if args.make_video:
        video_file_path = f'visuals/{args.env}/expert/{args.env}_expert_ep{cur_episode_id}.mp4'
        video_writer = hviz_utils.get_fast_video_writer(video_file_path, fps=30)

    # Trajectory from the start to the object
    ee_waypoints = get_ee_waypoints(
        obs, end_position_key='relative_target_position_sensor',
        end_position_is_relative=True,
        offsets=np.array([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.05]
        ])
    )
    ee_traj = np.array(make_traj(ee_waypoints, num_steps=100, interp_mode='quadratic')) / ee_ctrl_lim # in habitat they multiple by ee_ctrl_lim so we have to devide by it to get desired ee trajectory
    grip_traj = [-1.] * len(ee_traj)
    grip_traj[-3:] = [1., 1., 1.] # close actions should grip
    sub_traj, _ = execute_traj(
        env,
        ee_traj=ee_traj,
        grip_traj=grip_traj,
        video_writer=video_writer
    )
    final_obs = sub_traj[-1]
    traj.extend(sub_traj)
    
    action_traj = np.concatenate((ee_traj, np.expand_dims(grip_traj, axis=1)), axis=1)

    # # Trajectory from object to the goal location
    # ee_waypoints = get_ee_waypoints(
    #     final_obs, end_position_key='relative_object_to_goal_position_sensor',
    #     end_position_is_relative=True,
    #     offsets=np.array([
    #         [0., 0., 0.],
    #         [0., 0., 0.3], # Make the midpoint a bit above the table
    #         [0., 0., 0.05]
    #     ])
    # )
    # ee_traj = make_traj(ee_waypoints, num_steps=150, interp_mode='quadratic')
    # grip_traj = [1.] * len(ee_traj)
    # grip_traj[-1] = -1. # Last action should release
    # final_obs = execute_traj(
    #     env,
    #     ee_traj=ee_traj,
    #     grip_traj=grip_traj,
    #     video_writer=video_writer
    # )

    # And finally, from the goal location to the EE resting position
    ee_waypoints = get_ee_waypoints(
        final_obs, end_position_key='relative_resting_position',
        end_position_is_relative=True
    )
    ee_traj = np.array(make_traj(ee_waypoints, num_steps=100, interp_mode='quadratic')) / ee_ctrl_lim
    grip_traj = [1.] * len(ee_traj)
    sub_traj, last_info = execute_traj(
        env,
        ee_traj=ee_traj,
        grip_traj=grip_traj,
        video_writer=video_writer
    )

    success = last_info and last_info['rearrangepick_success']
    if success == True:
        traj.extend(sub_traj)
        actions = np.concatenate((ee_traj, np.expand_dims(grip_traj, axis=1)), axis=1)
        action_traj = np.concatenate((action_traj, actions), axis=0)

        save_traj(
            traj,
            action_traj=action_traj,
            episode_id=env.current_episode.episode_id,
            expert_traj_dir=expert_traj_dir
        )
    else:
        print(f'Failed episodeid={cur_episode_id}')

    if args.make_video:
        video_writer.close()
    
    return success


def generate_trajectories(args):
    assert args.env in env_configs, "Please choose a valid environment name"

    expert_traj_dir = os.path.join(EXPERT_TRAJ_BASE_DIR, args.env)
    assert not os.path.isdir(expert_traj_dir) or len(os.listdir(expert_traj_dir)) == 0, "Tried to overwrite existing trajectories. Please manually delete the trajectory dir"

    if not os.path.isdir(expert_traj_dir):
        os.mkdir(expert_traj_dir)

    env = make_IK_env(args)
    failed_episode_ids = []
    for ep_iter in range(len(env.episodes)):
        success = generate_single_trajectory(env, args, ep_iter, expert_traj_dir)
        if not success:
            cur_episode_id = env.current_episode.episode_id
            failed_episode_ids.append(cur_episode_id)
    
    if len(failed_episode_ids) > 0:
        print(f'Failed episode IDs: {failed_episode_ids}. Trying them again')
    else:
        return

    # Sometimes due to randomness in the trajectory generation, episodes fail. So just retry them
    #failed_episode_ids = ['290', '483']
    failed_episodes = [ep for ep in env.episodes if ep.episode_id in failed_episode_ids]
    while len(failed_episodes):
        env._episode_iterator.episodes = failed_episodes
        env.episodes = failed_episodes
        env._current_episode = failed_episodes[0]
        success = generate_single_trajectory(env, args, expert_traj_dir=expert_traj_dir)
        if not success:
            failed_episodes.append(env.current_episode)
        else:
            idx = [i for i, ep in enumerate(failed_episodes) if ep.episode_id == env.current_episode.episode_id][0]
            del failed_episodes[idx]


def make_videos_from_offline_trajectories():
    expert_traj_dir = os.path.join(EXPERT_TRAJ_BASE_DIR, 'pick_fruit')
    assert os.path.isdir(expert_traj_dir), f'Could not find directory {expert_traj_dir}'

    num_eps = len(os.listdir(expert_traj_dir))
    ep_ids = [int(ep.split('=')[-1]) for ep in os.listdir(expert_traj_dir)]
    for i in range(500):
        if i not in ep_ids:
            print(i)

    print(v)
    print(f'num eps: {num_eps}')
    ids = []
    for traj_dir in os.listdir(expert_traj_dir):
        ep_id = int(traj_dir.split('=')[-1])
        ids.append(ep_id)
    
    total_ids = list(range(num_eps))
    not_here = [id_ for id_ in total_ids if id_ not in ids]
    print(f'absent = {not_here}')


def main():
    os.environ['HABITAT_SIM_LOG'] = 'quiet'
    args = get_args()
    if args.make_videos_from_data:
        make_videos_from_offline_trajectories()
    else:
        generate_trajectories(args)
    


if __name__ == '__main__':
    main()
