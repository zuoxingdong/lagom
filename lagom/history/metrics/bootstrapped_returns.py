import numpy as np
import torch

from lagom.history import Trajectory
from lagom.history import Segment

from lagom.transform import ExpFactorCumSum


def bootstrapped_returns_from_trajectory(trajectory, V_last, gamma=1.0):
    r"""Return a list of (discounted) accumulated returns with bootstrapping for all 
    time steps, from a trajectory. 
    
    Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
    .. math::
        Q_t = r_t + \gamma r_{t+1} + \dots + \gamma^{T - t} r_T + \gamma^{T - t + 1} V(s_{T+1})
        
    .. note::

        The state value for terminal state is set as zero !
    
    Args:
        trajectory (Trajectory): a trajectory
        V_last (object): the value of the final state in the trajectory
        gamma (float): discounted factor
        
    Returns
    -------
    out : list
        a list of (discounted) bootstrapped returns
    """
    assert isinstance(trajectory, Trajectory) 
    
    if torch.is_tensor(V_last):
        V_last = V_last.item()
    if isinstance(V_last, np.ndarray):
        V_last = V_last.item()
        
    if trajectory.complete:
        V_last = 0.0
        
    out = ExpFactorCumSum(gamma)(trajectory.all_r + [V_last])
    out = out[:-1]  # last one is just state value itself
    
    return out
    
    
def bootstrapped_returns_from_segment(segment, all_V_last, gamma=1.0):
    r"""Return a list of (discounted) accumulated returns with bootstrapping for all 
    time steps, from a segment. 
    
    Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
    .. math::
        Q_t = r_t + \gamma r_{t+1} + \dots + \gamma^{T - t} r_T + \gamma^{T - t + 1} V(s_{T+1})
        
    .. note::

        The state value for terminal state is set as zero !
    
    Args:
        segment (Segment): a segment
        all_V_last (object): the value of all final states for each trajectory in the segment. 
        gamma (float): discounted factor
        
    Returns
    -------
    out : list
        a list of (discounted) bootstrapped returns
    """
    assert isinstance(segment, Segment)
    assert len(segment.trajectories) == len(all_V_last)
    
    out = []
    for trajectory, V_last in zip(segment.trajectories, all_V_last):
        out += bootstrapped_returns_from_trajectory(trajectory, V_last, gamma)
        
    return out
