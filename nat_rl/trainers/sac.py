import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from nat_rl.envs import HabitatArmActionWrapper
from nat_rl.trainers import BaseTrainer
from nat_rl.trainers.callbacks import VideoRecorderCallback


def make_env(env_class, env_kwargs):
    """
    Utility function for multiprocessed env.

    """
    def _init():
        env = env_class(**env_kwargs)
        env = HabitatArmActionWrapper(env)
        return env

    return _init


class SACTrainer(BaseTrainer):
    def __init__(self,
        env_class, env_kwargs=dict(),
        lr=1e-4, act_sigma=0.05, learning_starts=1e4,
        logdir='tb'
        ):
        num_cpu = 4
        env = SubprocVecEnv([make_env(env_class, env_kwargs) for _ in range(num_cpu)])
        env = VecMonitor(env)
        self.env = env
        self.algo = SAC(
            'MultiInputPolicy',
            self.env,
            buffer_size=int(1e5),
            learning_starts=learning_starts,
            action_noise=OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(self.env.action_space.shape),
                sigma=np.ones(self.env.action_space.shape) * act_sigma
            ),
            ent_coef='auto_0.1',
            verbose=2,
            tensorboard_log=logdir
        )

    def learn(self, timesteps):
        video_callback = VideoRecorderCallback(render_freq=100000)
        self.algo.learn(
            timesteps,
            log_interval=1,
            #eval_freq=5,
            n_eval_episodes=10,
            callback=video_callback
        )
