from turtle import end_fill
import numpy as np


def get_ee_waypoints(
    obs,
    end_position_key,
    end_position_is_relative=False,
    n_waypoints=3,
    offsets=None,
    add_noise=True
):
    assert 'ee_pos' in obs.keys(), 'EE position needs to be in the observation to gather waypoints'
    if not end_position_is_relative:
        assert end_position_key in obs.keys(), f'{end_position_key} needs to be in the observation'

    if offsets is not None:
        assert len(offsets) == n_waypoints, 'Offsets must be length n_waypoints (one offset array for start, each intermediate waypoint, and end)'

    start_pos = obs['ee_pos']
    if end_position_is_relative:
        end_pos = start_pos + obs[end_position_key]
    else:
        end_pos = obs[end_position_key]
    
    midpoint_pos = (start_pos + end_pos) / 2
    
    if add_noise:
        midpoint_pos += np.random.normal(loc=0.0, scale=0.05)


    ee_waypoints = np.array([
        start_pos,
        midpoint_pos,
        end_pos
    ])
    if offsets is not None:
        ee_waypoints += offsets

    return ee_waypoints