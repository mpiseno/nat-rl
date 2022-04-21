import os
import argparse

import gym
import numpy as np
import habitat

from stanford_habitat.measures import * # register
from nat_rl.utils.env_utils import make_habitat_pick_single_object_env, make_habitat_GC_pick_single_object_env
from habitat_sim.utils import viz_utils as vut
from habitat_baselines.utils.render_wrapper import overlay_frame



def main():
    os.environ['HABITAT_SIM_LOG'] = 'quiet'
    
    env = make_habitat_pick_single_object_env()
    obs = env.reset()
    import pdb; pdb.set_trace()

    print('yo')



if __name__ == '__main__':
    main()
