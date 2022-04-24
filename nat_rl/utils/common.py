import os
from pathlib import Path

import numpy as np

from habitat_sim.utils import viz_utils as vut


def count_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_count = sum([np.prod(p.size()) for p in model_parameters])
    return param_count
