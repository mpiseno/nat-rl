from collections import OrderedDict

import gym
import numpy as np


class HabitatArmActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Dict), \
            "Should only be used to wrap Dict envs."

        self.env = env
        self.dict_action_keys = self.env.action_space['ARM_ACTION'].spaces.keys()

        action_space_dim = 8 # 7 arm actions, 1 gripper action
        self.action_space_dim = action_space_dim
        self.action_space = gym.spaces.Box(-1, 1, (self.action_space_dim,))

    def action(self, act_array):
        '''
        act_array (np.ndarray): The action output from the policy. Need to convert it to a dictionary so that Habitat environments can handle it
        '''
        arm_action = act_array[:7].copy().astype(np.float32)
        grip_action = act_array[-1:].copy().astype(np.float32)
        new_act = {
            'action': 'ARM_ACTION',
            'action_args': OrderedDict([
                ('arm_action', arm_action),
                ('grip_action', grip_action)
            ])
        }
        return new_act