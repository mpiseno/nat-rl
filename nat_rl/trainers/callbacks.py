import gym
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Dict

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class VideoRecorderCallback(BaseCallback):
    def __init__(self, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = self.locals.get('env')
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                eval_env = _locals.get('env')
                screen = eval_env.render(mode="robot_third_rgb")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            eval_env = self.locals.get('env')
            evaluate_policy(
                self.model,
                eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


# class HabitatInfoCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(HabitatInfoCallback, self).__init__(verbose)
    
#     def _on_step(self) -> bool:
#         import pdb; pdb.set_trace()
#         log_interval = self.locals.get('log_interval', None)
#         ep_num = self.locals.get('self')._episide_number
#         if log_interval and ep_num % log_interval == 0:
#             did_pick = safe_mean([
#                 ep_info['did_pick_object']
#                 self.locals.get('self').ep_info_buffer
#             ])
#         self.logger.record('info/did_pick', did_pick)
#         return True
