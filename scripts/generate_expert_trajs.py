import os
import json
import gzip
import argparse
import random
from threading import current_thread

import gym
import numpy as np
import habitat
import torch
import clip

from PIL import Image
from habitat_sim.utils import viz_utils as hviz_utils
from habitat.utils.visualizations.utils import observations_to_image

from stanford_habitat.utils import make_traj, execute_traj
from stanford_habitat.measures import * # register

from nat_rl.models import load_clip_model
from nat_rl.utils.ik import get_ee_waypoints
from nat_rl.utils.common import expert_img_sort_fn
from nat_rl.utils.env_utils import PICK_SINGLE_OBJECT_CONFIG, PICK_FRUIT_CONFIG, SPATIAL_REASONING_CONFIG, insert_test_dataset

'''
Uses IK to solve pick&place tasks and saves the trajectory buffer
'''

EXPERT_TRAJ_BASE_DIR = 'data/expert_trajs/'
env_configs = {
    'pick_single_object-v0': PICK_SINGLE_OBJECT_CONFIG,
    'pick_fruit': PICK_FRUIT_CONFIG,
    'pick_fruit_test': PICK_FRUIT_CONFIG,
    'spatial_reasoning': SPATIAL_REASONING_CONFIG,
    'spatial_reasoning_test': SPATIAL_REASONING_CONFIG
}
dataset_files = {
    'pick_fruit': 'data/pick_datasets/pick_fruit/pick_fruit.json copy.gz',
    'pick_fruit_test': 'data/pick_datasets/pick_fruit/pick_fruit_test.json.gz'
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)

    parser.add_argument('--make_video', action='store_true', default=False)

    parser.add_argument('--generate_image_trajs', action='store_true', default=False)
    parser.add_argument('--generate_clip_img_embeddings', action='store_true', default=False)
    parser.add_argument('--generate_clip_lang_embeddings', action='store_true', default=False)
    parser.add_argument('--pretrained_clip_path', type=str, default=None)

    args = parser.parse_args()
    return args


def is_pick_and_place_env(env_name):
    return env_name in ['spatial_reasoning', 'spatial_reasoning_test']


def make_IK_config(config, args):
    config.defrost()

    config.TASK.ACTIONS.ARM_ACTION.ARM_CONTROLLER = "ArmEEAction"
    config.TASK.SENSORS = [
        "RELATIVE_TARGET_POSITION_SENSOR",
        "JOINT_SENSOR",
        "IS_HOLDING_SENSOR",
        "END_EFFECTOR_SENSOR",
        "RELATIVE_RESTING_POS_SENSOR"
    ]

    if is_pick_and_place_env(args.env):
        config.TASK.SENSORS.append("RELATIVE_OBJECT_TO_GOAL_POS_SENSOR")
    
    config.freeze()
    return config


def make_IK_env(args):
    config_path = env_configs[args.env]
    config_path = os.path.join(os.getcwd(), config_path)

    env_config = habitat.get_config(config_path)
    env_config = make_IK_config(env_config, args)
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


def make_video(video_file_path, traj):
    video_writer = hviz_utils.get_fast_video_writer(video_file_path, fps=30)
    for obs in traj:
        video_writer.append_data(obs['robot_third_rgb'])
    
    video_writer.close()


def pick_up_target(env, start_obs, ee_ctrl_lim):
    # Trajectory from the start to the object
    ee_waypoints = get_ee_waypoints(
        start_obs, end_position_key='relative_target_position_sensor',
        end_position_is_relative=True,
        offsets=np.array([
            [0., 0., 0.],
            [0., 0., 0.3],
            [0., 0., 0.05]
        ])
    )
    ee_traj = np.array(
        make_traj(ee_waypoints, num_steps=100, interp_mode='quadratic')
    ) / ee_ctrl_lim # in habitat they multiple by ee_ctrl_lim so we have to devide by it to get desired ee trajectory
    grip_traj = [-1.] * len(ee_traj)
    grip_traj[-3:] = [1., 1., 1.] # close actions should grip
    sub_traj, _ = execute_traj(
        env,
        ee_traj=ee_traj,
        grip_traj=grip_traj,
    )
    
    actions = np.concatenate((ee_traj, np.expand_dims(grip_traj, axis=1)), axis=1)
    return sub_traj, actions, _


def move_target_to_goal(env, start_obs, ee_ctrl_lim):
    # Trajectory from object to the goal location
    ee_waypoints = get_ee_waypoints(
        start_obs, end_position_key='relative_object_to_goal_position_sensor',
        end_position_is_relative=True,
        offsets=np.array([
            [0., 0., 0.],
            [0., 0., 0.3], # Make the midpoint a bit above the table
            [0., 0., 0.05]
        ])
    )
    ee_traj = np.array(make_traj(ee_waypoints, num_steps=100, interp_mode='quadratic')) / ee_ctrl_lim
    grip_traj = [1.] * len(ee_traj)
    grip_traj[-3:] = [-1., -1., -1.] # Last few actions should release
    sub_traj, _ = execute_traj(
        env,
        ee_traj=ee_traj,
        grip_traj=grip_traj,
    )    

    actions = np.concatenate((ee_traj, np.expand_dims(grip_traj, axis=1)), axis=1)
    return sub_traj, actions, _


def move_ee_to_rest(env, start_obs, ee_ctrl_lim):
    # And finally, from the goal location to the EE resting position
    ee_waypoints = get_ee_waypoints(
        start_obs, end_position_key='relative_resting_position',
        end_position_is_relative=True
    )
    ee_traj = np.array(make_traj(ee_waypoints, num_steps=100, interp_mode='quadratic')) / ee_ctrl_lim
    grip_traj = [-1.] * len(ee_traj)
    sub_traj, last_info = execute_traj(
        env,
        ee_traj=ee_traj,
        grip_traj=grip_traj,
    )

    actions = np.concatenate((ee_traj, np.expand_dims(grip_traj, axis=1)), axis=1)
    return sub_traj, actions, last_info


def generate_single_trajectory(env, args, ep_iter=-1, expert_traj_dir=None):
    video_writer = None
    ee_ctrl_lim = env._config.TASK.ACTIONS.ARM_ACTION.EE_CTRL_LIM
    obs = env.reset()
    cur_episode_id = env.current_episode.episode_id
    print(f'Generating episodeid={cur_episode_id} ({ep_iter + 1}/{len(env.episodes)})')

    def rollout(obs0, sub_traj_fns):
        traj = []
        traj.append(obs0)
        action_traj = np.array([[-1, -1, -1, -1]])
        for fn in sub_traj_fns:
            init_obs = traj[-1]
            sub_traj, sub_action_traj, info = fn(env, init_obs, ee_ctrl_lim)

            traj.extend(sub_traj)
            action_traj = np.concatenate((action_traj, sub_action_traj), axis=0)
        
        return traj, action_traj[1:], info

    sub_traj_fns = []
    if is_pick_and_place_env(args.env):
        sub_traj_fns.extend([pick_up_target, move_target_to_goal, move_ee_to_rest])
    else:
        sub_traj_fns.extend([pick_up_target, move_ee_to_rest])

    traj, action_traj, final_info = rollout(obs, sub_traj_fns)
    success = final_info and (
        final_info.get('rearrangepickplace_success', False)
        or final_info.get('rearrangepick_success', False)
    )

    if success:
        save_traj(
            traj,
            action_traj=action_traj,
            episode_id=env.current_episode.episode_id,
            expert_traj_dir=expert_traj_dir
        )

    if args.make_video:
        video_file_path = f'visuals/expert/{args.env}_expert_ep{cur_episode_id}.mp4'
        make_video(video_file_path, traj)
    
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
    failed_episodes = [ep for ep in env.episodes if ep.episode_id in failed_episode_ids]
    env._episode_iterator.episodes = failed_episodes
    env.episodes = failed_episodes
    env._current_episode = failed_episodes[0]
    failed_again_episodes = []
    for ep in failed_episodes:
        success = generate_single_trajectory(env, args, expert_traj_dir=expert_traj_dir)
        if not success:
            failed_again_episodes.append(env.current_episode)
    
    failed_again_episode_ids = [ep.episode_id for ep in failed_again_episodes]
    if len(failed_again_episode_ids) > 0:
        print(f'''
            Failed episodes after a second try (Please manually removed or replace these episodes from the <dataset>.json.gz file, because they do not have corresponding expert trajectories or goal images):\n
            {failed_again_episode_ids}
        ''')


def generate_clip_embeddings_single_traj(episode_dir, model, preprocess, device):
    episode_files = os.listdir(episode_dir)
    obs_img_files = sorted(filter(lambda x: x.endswith('.png'), episode_files), key=expert_img_sort_fn)

    images = [
        np.array(
            Image.open(os.path.join(episode_dir, img_file)),
            dtype=np.uint8
        )
        for img_file in obs_img_files
    ]
    images = np.stack(images, axis=0) / 255 # the ToTensor transform usually normalizes to [0, 1] but is not compatable with batched images
    images = torch.as_tensor(np.transpose(images, (0, 3, 1, 2)), dtype=torch.float32)
    images = preprocess(images)
    with torch.no_grad():
        image_embeddings = model.encode_image(images.to(device))
    
    image_embeddings = image_embeddings.cpu().numpy()

    embeddings_fname = os.path.join(episode_dir, 'clip_embeddings.npy')
    np.save(embeddings_fname, image_embeddings.astype(np.float32))


def generate_clip_embeddings(args, lang=False):
    if args.pretrained_clip_path is not None:
        pass

    expert_traj_dir = os.path.join(EXPERT_TRAJ_BASE_DIR, args.env)
    episode_dirs = os.listdir(expert_traj_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip_model(keep_original_transforms=False, device=device)
    print(f'Using CLIP model: {model}\nUsing preprocess: {preprocess}')

    print(f'Generating CLIP Image embeddings from: {expert_traj_dir} | using device: {device}')
    for i, episode_dir in enumerate(episode_dirs):
        print(f'Generating CLIP embeddings {(i + 1)} / {len(episode_dirs)}')
        generate_clip_embeddings_single_traj(
            os.path.join(expert_traj_dir, episode_dir),
            model,
            preprocess,
            device,
        )


def get_episode_id_to_target_object(dataset_file):
    print(f'Parsing {dataset_file} for object info')

    ep_id_to_obj = {}
    with gzip.open(dataset_file, 'rb') as f:
        episodes = json.loads(f.read())['episodes']
        for ep in episodes:
            ep_id = int(ep['episode_id'])
            obj_handle = list(ep['info']['object_labels'].keys())[0]
            obj = obj_handle.split('_')[1]
            ep_id_to_obj[ep_id] = obj
    
    return ep_id_to_obj


def generate_clip_language_embeddings_single_traj(
    episode_dir,
    model,
    device,
    true_object,
    generate_language_fn
):
    utterance = generate_language_fn(true_object)
    token = clip.tokenize([utterance])
    text_embedding = model.encode_text(token).detach().numpy()

    embedding_fname = os.path.join(episode_dir, 'clip_language_embeddings.npy')
    np.save(embedding_fname, text_embedding.astype(np.float32))
    with open(os.path.join(episode_dir, 'raw_language.txt'), 'w') as f:
        f.write(utterance)


def generate_language_pick_fruit(true_object):
    obj_to_adjective = {
        'apple': 'red',
        'orange': 'orange',
        'plum': 'purple',
        'banana': 'yellow'
    }
    noun1 = random.choice(['image', 'picture', 'photo'])
    verb1 = random.choice(['holding', 'picking up'])
    a_vs_an1 = 'A' if noun1 in ['picture', 'photo'] else 'An'
    adjective = obj_to_adjective[true_object]
    a_vs_an2 = 'a' if adjective in ['red', 'purple', 'yellow'] else 'an'
    utterance = f'{a_vs_an1} {noun1} of a robot {verb1} {a_vs_an2} {adjective} {true_object} above the table.'
    return utterance


def generate_language_spatial_reasoning(true_object):
    pass


generate_language_fns = {
    'pick_fruit': generate_language_pick_fruit,
    'pick_fruit_test': generate_language_pick_fruit,
    'spatial_reasoning': generate_language_spatial_reasoning,
    'spatial_reasoning_test': generate_language_spatial_reasoning
}


def generate_clip_lang_embeddings(args):
    if args.pretrained_clip_path is not None:
        pass

    expert_traj_dir = os.path.join(EXPERT_TRAJ_BASE_DIR, args.env)
    episode_dirs = os.listdir(expert_traj_dir)

    device = "cpu"
    model, _ = load_clip_model(keep_original_transforms=False, device=device)

    dataset_file = dataset_files[args.env]
    ep_id_to_obj = get_episode_id_to_target_object(dataset_file)
    generate_language_fn = generate_language_fns[args.env]

    print(f'Generating CLIP Language embeddings from: {expert_traj_dir} | using device: {device}')
    for i, episode_dir in enumerate(episode_dirs):
        print(f'Generating CLIP Language embeddings {(i + 1)} / {len(episode_dirs)}')
        ep_id = int(episode_dir.split('=')[-1])
        generate_clip_language_embeddings_single_traj(
            os.path.join(expert_traj_dir, episode_dir),
            model,
            device,
            ep_id_to_obj[ep_id],
            generate_language_fn
        )


def main():
    os.environ['HABITAT_SIM_LOG'] = 'quiet'
    args = get_args()
    
    # Generates the actual trajectories and saves image observations
    if args.generate_image_trajs:
        generate_trajectories(args)
    
    # Saves CLIP embeddings of the generated images. Assumes the images are already saved
    if args.generate_clip_img_embeddings:
        generate_clip_embeddings(args)

    if args.generate_clip_lang_embeddings:
        generate_clip_lang_embeddings(args)


if __name__ == '__main__':
    main()
