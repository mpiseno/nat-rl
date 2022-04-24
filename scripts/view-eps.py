import os
import pdb

import gym
import habitat
import habitat_baselines.utils.gym_definitions as habitat_gym

from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.render_wrapper import overlay_frame
from habitat_sim.utils import viz_utils as vut

from nat_rl.utils.env_utils import make_habitat_pick_single_object_env, make_pick_fruit_env


EPS_TO_VIEW = 10


def main():
    env = make_pick_fruit_env()

    # import collections
    # counter = collections.defaultdict(int)
    # for episode in env.episodes:
    #     target = list(episode.targets.keys())[0]
    #     counter[target] += 1

    # import pdb; pdb.set_trace()

    for i in range(EPS_TO_VIEW):
        print(f"Agent acting inside environment | Episode {i}")
        obs = env.reset()
        count_steps = 0

        # To save the video
        video_file_path = f"visuals/tmp/episode{i}.mp4"
        video_writer = vut.get_fast_video_writer(video_file_path, fps=30)
        render_obs = obs['robot_third_rgb']
        video_writer.append_data(render_obs)
        while not env.habitat_env.episode_over:
            action = env.action_space.sample()
            obs, r, done, info = env.step(action)  # noqa: F841
            render_obs = obs['robot_third_rgb']
            
            #render_obs = observations_to_image(obs, info)
        
            
            #render_obs = overlay_frame(render_obs, {'num objs': len(env._current_episode.rigid_objs)})
            video_writer.append_data(render_obs)

            count_steps += 1

        print(f"Episode {i} finished after {count_steps} steps.")
        print(f"Dist to goal: {info['object_to_goal_distance']}")

        video_writer.close()


if __name__ == '__main__':
    main()
