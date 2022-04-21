import os
import PIL

import numpy as np

from imitation.data.rollout import *
from imitation.data.types import Trajectory


def goal_img_sort_fn(fname):
    key = ''.join([c if c.isdigit() else '' for c in fname])
    key = int(key) if key else -1
    return key


def generate_offline_trajectories(expert_traj_dir, env):
    def generate_single_trajectory(ep_dir, env):
        actions_fname = os.path.join(ep_dir, 'actions.npy')
        actions = np.load(actions_fname)

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
        
        goal_imgs = [goal_img.copy() for _ in range(len(images))]
        obs = [
            {
                'robot_third_rgb': img,
                #'goal': goal_img.copy()
            }
            for img in images
        ]
        
        out_dict = {}
        out_dict['obs'] = np.stack(obs, axis=0)
        out_dict['acts'] = actions
        #out_dict['rews'] = np.zeros((len(actions),), dtype=np.float32)
        out_dict['infos'] = None

        new_traj = Trajectory(**out_dict, terminal=True)
        assert new_traj.acts.shape[0] == new_traj.obs.shape[0] - 1
        return new_traj

    trajs = []
    episode_dirs = os.listdir(expert_traj_dir)
    for episode_dir in episode_dirs:
        episode_dir = os.path.join(expert_traj_dir, episode_dir)
        trajs.append(generate_single_trajectory(episode_dir, env))
    
    return trajs


def make_rollouts_from_offline_data(expert_traj_dir, env):
    trajs = generate_offline_trajectories(expert_traj_dir, env)
    return trajs
