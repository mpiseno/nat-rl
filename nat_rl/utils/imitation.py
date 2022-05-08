import os
import PIL

import numpy as np

from imitation.data.rollout import *
from imitation.data.types import Trajectory


def goal_img_sort_fn(fname):
    key = ''.join([c if c.isdigit() else '' for c in fname])
    key = int(key) if key else -1
    return key


def get_offline_images(ep_dir):
    # determine goal image
    img_fnames = sorted(os.listdir(ep_dir), key=goal_img_sort_fn)
    last_obs_fname = img_fnames[-1]
    last_obs_fname = os.path.join(ep_dir, last_obs_fname)
    goal_img = np.array(PIL.Image.open(last_obs_fname))

    images = []
    for fname in img_fnames:
        if fname.endswith('.png'):
            image_fname = os.path.join(ep_dir, fname)
            image = np.array(PIL.Image.open(image_fname))
            images.append(image)
    
    return images, goal_img


def generate_single_trajectory_raw_images(ep_dir):
    actions_fname = os.path.join(ep_dir, 'actions.npy')
    actions = np.load(actions_fname)

    images, goal_img = get_offline_images(ep_dir)
    obs = [
        {
            'robot_third_rgb': img,
            'goal': goal_img.copy()
        }
        for img in images
    ]
    
    out_dict = {}
    out_dict['obs'] = np.stack(obs, axis=0)
    out_dict['acts'] = actions
    out_dict['infos'] = None

    new_traj = Trajectory(**out_dict, terminal=True)
    assert new_traj.acts.shape[0] == new_traj.obs.shape[0] - 1
    return new_traj


def generate_single_trajectory_clip_embeddings(ep_dir, embed_goal_only):
    actions_fname = os.path.join(ep_dir, 'actions.npy')
    actions = np.load(actions_fname)

    clip_embeddings_fname = os.path.join(ep_dir, 'clip_embeddings.npy')
    clip_embeddings = np.load(clip_embeddings_fname)
    goal_embedding = clip_embeddings[-1]


    if embed_goal_only == True:
        images, _ = get_offline_images(ep_dir)
        obs = [
            {
                'robot_third_rgb': img,
                'goal': goal_embedding.copy()
            }
            for img in images
        ]
    else:
        obs = [
            {
                'robot_third_rgb': embedding,
                'goal': goal_embedding.copy()
            }
            for embedding in clip_embeddings
        ]
    
    out_dict = {
        'obs': np.stack(obs, axis=0),
        'acts': actions,
        'infos': None
    }

    new_traj = Trajectory(**out_dict, terminal=True)
    assert new_traj.acts.shape[0] == new_traj.obs.shape[0] - 1
    return new_traj


def generate_offline_trajectories(expert_traj_dir, clip, embed_goal_only=True):
    trajs = []
    episode_dirs = os.listdir(expert_traj_dir)
    for episode_dir in episode_dirs:
        episode_dir = os.path.join(expert_traj_dir, episode_dir)
        if clip == True:
            traj = generate_single_trajectory_clip_embeddings(episode_dir, embed_goal_only)
        else:
            traj = generate_single_trajectory_raw_images(episode_dir)

        trajs.append(traj)
    
    return trajs


def generate_single_dummy_trajectory():
    obs = [
        {
            'robot_third_rgb': np.random.randint(low=0, high=256, size=(128, 128, 3)),
            'goal':  np.random.normal(size=(512,)) #np.random.randint(low=0, high=256, size=(128, 128, 3)),
        }
        for _ in range(200)
    ]
    
    out_dict = {
        'obs': np.stack(obs, axis=0),
        'acts': np.random.normal(size=(199, 4)),
        'infos': None
    }

    new_traj = Trajectory(**out_dict, terminal=True)
    assert new_traj.acts.shape[0] == new_traj.obs.shape[0] - 1
    return new_traj


def generate_dummy_trajectories(n_trajs=50):
    trajs = []
    for _ in range(n_trajs):
        traj = generate_single_dummy_trajectory()
        trajs.append(traj)
    
    return trajs