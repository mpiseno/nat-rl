import os
import pdb

from pathlib import Path

import gym
import habitat
import habitat_baselines.utils.gym_definitions as habitat_gym

from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.render_wrapper import overlay_frame
from habitat_sim.utils import viz_utils as vut

from stanford_habitat.envs import *
from stanford_habitat.measures import *
from stanford_habitat.tasks import *


CONFIG = 'configs/test/test_install.yaml'


def main():
    config = habitat.get_config(CONFIG)
    env = habitat.Env(config=config)
    visuals_dir = 'visuals/view_eps/'
    Path(visuals_dir).mkdir(parents=True, exist_ok=True)
    print(f"Agent acting inside environment")
    obs = env.reset()
    count_steps = 0

    # To save the video
    video_file_path = os.path.join(visuals_dir, f'test_episode.mp4')
    video_writer = vut.get_fast_video_writer(video_file_path, fps=30)
    render_obs = obs['robot_third_rgb']
    video_writer.append_data(render_obs)
    while not env.episode_over:
        action = env.action_space.sample()
        obs = env.step(action)  # noqa: F841
        render_obs = obs['robot_third_rgb']
        video_writer.append_data(render_obs)

        count_steps += 1

    print(f"Episode finished after {count_steps} steps.")
    video_writer.close()


if __name__ == '__main__':
    os.environ['HABITAT_SIM_LOG'] = 'quiet'
    main()
