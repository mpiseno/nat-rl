import os
import pdb

import gym
import habitat
import habitat_baselines.utils.gym_definitions as habitat_gym

from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.render_wrapper import overlay_frame
from habitat_sim.utils import viz_utils as vut


EPS_TO_VIEW = 10
CONFIG_PATH = "configs/pick-table.yaml"


def insert_render_options(config):
    config.defrost()
    config.SIMULATOR.THIRD_RGB_SENSOR.WIDTH = 512
    config.SIMULATOR.THIRD_RGB_SENSOR.HEIGHT = 512
    config.SIMULATOR.AGENT_0.SENSORS.append("THIRD_RGB_SENSOR")
    config.freeze()
    return config


def main():
    config=insert_render_options(
            habitat.get_config(
                CONFIG_PATH
            )
        )

    env = habitat.Env(config=config)
    obs = env.reset()

    for i in range(EPS_TO_VIEW):
        env.reset()

        print(f"Agent acting inside environment | Episode {i}")
        count_steps = 0

        # To save the video
        print(f'EP: {i} | num objs: {len(env._current_episode.rigid_objs)}')
        video_file_path = f"visuals/episode{i}.mp4"
        video_writer = vut.get_fast_video_writer(video_file_path, fps=30)
        while not env.episode_over:
            action = env.action_space.sample()
            obs = env.step(action)  # noqa: F841
            info = env.get_metrics()
            render_obs = obs['robot_third_rgb']
            #render_obs = observations_to_image(obs, info)
            #import pdb; pdb.set_trace()
            
            #render_obs = overlay_frame(render_obs, {'num objs': len(env._current_episode.rigid_objs)})
            video_writer.append_data(render_obs)

            count_steps += 1

        print(f"Episode {i} finished after {count_steps} steps.")
        print(f"Dist to goal: {info['object_to_goal_distance']}")

        video_writer.close()


if __name__ == '__main__':
    main()
