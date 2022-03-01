import numpy as np
import habitat

from gym import spaces


class PickPlaceTableEnv(habitat.RLEnv):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._reward_measure_name = self.config.TASK.REWARD_MEASUREMENT
        self._success_measure_name = self.config.TASK.SUCCESS_MEASUREMENT

        # Need to convert depths images to uint8 in range [0, 255] for compatibility with stable baselines
        for obs_name, subspace in self.observation_space.spaces.items():
            if 'depth' in obs_name.lower():
                low = np.zeros(subspace.low.shape, dtype=np.uint8)
                high = np.ones(subspace.high.shape, dtype=np.uint8) * 255
                self.observation_space.spaces[obs_name] = spaces.Box(
                    low=low, high=high, dtype=np.uint8
                )

    def _convert_depth(self, obs):
        for obs_name, value in obs.items():
            if 'depth' in obs_name.lower():
                depth_obs = ((value / np.max(value)) * 255).astype(np.uint8)
                obs[obs_name] = depth_obs
        
        return obs

    def reset(self):
        obs = super().reset()
        obs = self._convert_depth(obs)
        return obs

    def step(self, action, **kwargs):
        obs, reward, done, info = super().step(action=action, **kwargs)
        obs = self._convert_depth(obs)
        return obs, reward, done, info

    def get_reward_range(self):
        # We don't know what the reward measure is bounded by
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        reward = self._env.get_metrics()[self._reward_measure_name]
        if self._episode_success():
            reward += self.config.RL.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()