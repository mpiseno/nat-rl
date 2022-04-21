import numpy as np
import torch as th


def maybe_transpose(obs):
    if type(obs) == dict:
        obs = {key: maybe_transpose(val) for key, val in obs.items()}
    else:
        if obs.shape[-1] == 3 and len(obs.shape) == 4:
            if type(obs) == np.ndarray:
                obs = obs.transpose(0, 3, 1, 2)
            elif type(obs) == th.Tensor:
                obs = th.permute(obs, (0, 3, 1, 2))
    
    return obs