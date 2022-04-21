import os
import argparse
from threading import current_thread

import gym
import numpy as np
import habitat

from stanford_habitat.measures import * # register
from nat_rl.utils.env_utils import make_habitat_pick_single_object_env, make_habitat_GC_pick_single_object_env
from habitat_sim.utils import viz_utils as vut
from habitat_baselines.utils.render_wrapper import overlay_frame



def main():
    os.environ['HABITAT_SIM_LOG'] = 'quiet'
    
    env = make_habitat_GC_pick_single_object_env()
    obs = env.reset()

    cur_ep = env.habitat_env.current_episode.episode_id
    actions = np.load(f'data/expert_trajs/episodeid={cur_ep}/actions.npy')
    
    video_file_path = f"visuals/act_check.mp4"
    video_writer = vut.get_fast_video_writer(video_file_path, fps=30)
    video_writer.append_data(obs['robot_third_rgb'])
    t = 0
    while True:
        action = actions[t]
        obs, r, d, info = env.step(action)

        render_obs = obs['robot_third_rgb']
        render_obs = overlay_frame(render_obs, {'success': info['rearrangepick_success']})
        video_writer.append_data(render_obs)

        if d:
            break

        t += 1
    
    video_writer.close()
    print(f'final t: {t}')


    #import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
