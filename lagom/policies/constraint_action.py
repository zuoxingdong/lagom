import numpy as np

import torch


def constraint_action(env_spec, action):
    r"""Clipping the action with valid upper/lower bound defined in action space. 

    .. note::

        We assume all dimensions in continuous action space share the identical high and low value
        e.g. low = [-2.0, -2.0] and high = [2.0, 2.0]

    .. warning::

        The constraint action should be placed after computing the log-probability. It happens before
        it, the log-probability will be definitely wrong value. 

    Args:
        action (Tensor): sampled action

    Returns
    -------
    constrained_action : Tensor
        constrained action.
    """
    low = np.unique(env_spec.action_space.low)
    high = np.unique(env_spec.action_space.high)
    assert low.ndim == 1 and high.ndim == 1, 'low and high should be identical for each dimension'
    assert -low.item() == high.item(), 'low and high should be identical with absolute value'

    # TODO: wait for PyTorch clamp to support multidimensional min/max
    constrained_action = torch.clamp(action, min=low.item(), max=high.item())

    return constrained_action
