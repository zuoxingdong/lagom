import numpy as np
import torch

from lagom.history import Trajectory
from lagom.history import Segment


def td0_target(list_r, list_V, gamma):
    r"""Calculate a list of TD(0) targets given a list of rewards, a list of state values and a discounted factor.
    
    Let :math:`r_1, r_2, \dots, r_T` be a list of rewards and let :math:`V(s_0), V(s_1), \dots, V(s_{T-1}), V(s_{T})`
    be a list of state values including a last state value. Let :math:`\gamma` be a discounted factor, 
    the TD(0) targets are calculated as follows
        
    .. math::
        r_t + \gamma V(s_t), \forall t = 1, 2, \dots, T
    
    Args:
        list_r (list): a list of rewards. 
        list_V (list): a list of state values including the last state value.
        gamma (float): a discounted factor. 
    
    Returns
    -------
    list_td0_target : list
        a list of TD(0) targets
    """
    assert isinstance(list_r, list) and isinstance(list_V, list)
    assert len(list_V) == len(list_r) + 1
    assert gamma >= 0.0 and gamma <= 1.0
    
    list_td0_target = [r + gamma*V for r, V in zip(list_r, list_V[1:])]
    
    return list_td0_target


def td0_target_from_trajectory(trajectory, Vs, V_last, gamma):
    r"""Return a list of TD(0) targets from a trajectory. 
    
    .. note::

        The state value for terminal state is set as zero !
    
    Args:
        trajectory (Trajectory): a trajectory
        Vs (list): a list of state values for each time step excluding the final state. 
        V_last (object): the value of the final state in the trajectory
        gamma (float): discounted factor
        
    Returns
    -------
    out : list
        a list of TD(0) targets
    """
    assert isinstance(trajectory, Trajectory)
    assert isinstance(Vs, list) and len(Vs) == trajectory.T
    
    def to_raw(x):
        if torch.is_tensor(x):
            return x.item()
        elif isinstance(x, np.ndarray):
            return x.item()
        else:
            return x
        
    Vs = [to_raw(V) for V in Vs]
    V_last = to_raw(V_last)
    
    if trajectory.complete:
        V_last = 0.0
        
    out = td0_target(trajectory.all_r, Vs + [V_last], gamma)
    
    return out


def td0_target_from_segment(segment, all_Vs, all_V_last, gamma):
    r"""Return a list of TD(0) targets from a segment. 
    
    .. note::

        The state value for terminal state is set as zero !
    
    Args:
        segment (Segment): a segment
        all_Vs (list): a list of state values for each time step and each inner trajectory
            excluding the final state. 
        all_V_last (list): a list of final state values for each inner trajectory. 
        gamma (float): discounted factor
        
    Returns
    -------
    out : list
        a list of TD(0) targets
    """
    assert isinstance(segment, Segment)
    assert isinstance(all_Vs, list) and len(all_Vs) == len(segment.trajectories)
    assert len(all_V_last) == len(segment.trajectories)
    
    out = []
    for trajectory, Vs, V_last in zip(segment.trajectories, all_Vs, all_V_last):
        out += td0_target_from_trajectory(trajectory, Vs, V_last, gamma)
    
    return out
    

def td0_error(list_r, list_V, gamma):
    r"""Calculate a list of TD(0) errors given a list of rewards, a list of state values and a discounted factor.
    
    Let :math:`r_1, r_2, \dots, r_T` be a list of rewards and let :math:`V(s_0), V(s_1), \dots, V(s_{T-1}), V(s_{T})`
    be a list of state values including a last state value. Let :math:`\gamma` be a discounted factor, 
    the TD(0) errors are calculated as follows
    
    .. math::
        \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
        
    Args:
        list_r (list): a list of rewards. 
        list_V (list): a list of state values including the last state value.
        gamma (float): a discounted factor. 
    
    Returns
    -------
    list_td0_error : list
        a list of TD(0) errors
    """
    assert isinstance(list_r, list) and isinstance(list_V, list)
    assert len(list_V) == len(list_r) + 1
    assert gamma >= 0.0 and gamma <= 1.0
    
    list_td0_target = td0_target(list_r, list_V, gamma)
    
    list_td0_error = [td0_target - V for td0_target, V in zip(list_td0_target, list_V[:-1])]
    
    return list_td0_error


def td0_error_from_trajectory(trajectory, Vs, V_last, gamma):
    r"""Return a list of TD(0) errors from a trajectory. 
    
    .. note::

        The state value for terminal state is set as zero !
    
    Args:
        trajectory (Trajectory): a trajectory
        Vs (list): a list of state values for each time step excluding the final state. 
        V_last (object): the value of the final state in the trajectory
        gamma (float): discounted factor
        
    Returns
    -------
    out : list
        a list of TD(0) errors
    """
    assert isinstance(trajectory, Trajectory)
    assert isinstance(Vs, list) and len(Vs) == trajectory.T
    
    def to_raw(x):
        if torch.is_tensor(x):
            return x.item()
        elif isinstance(x, np.ndarray):
            return x.item()
        else:
            return x
        
    Vs = [to_raw(V) for V in Vs]
    V_last = to_raw(V_last)
    
    if trajectory.complete:
        V_last = 0.0
        
    out = td0_error(trajectory.all_r, Vs + [V_last], gamma)
    
    return out


def td0_error_from_segment(segment, all_Vs, all_V_last, gamma):
    r"""Return a list of TD(0) errors from a segment. 
    
    .. note::

        The state value for terminal state is set as zero !
    
    Args:
        segment (Segment): a segment
        all_Vs (list): a list of state values for each time step and each inner trajectory
            excluding the final state. 
        all_V_last (list): a list of final state values for each inner trajectory. 
        gamma (float): discounted factor
        
    Returns
    -------
    out : list
        a list of TD(0) errors
    """
    assert isinstance(segment, Segment)
    assert isinstance(all_Vs, list) and len(all_Vs) == len(segment.trajectories)
    assert len(all_V_last) == len(segment.trajectories)
    
    out = []
    for trajectory, Vs, V_last in zip(segment.trajectories, all_Vs, all_V_last):
        out += td0_error_from_trajectory(trajectory, Vs, V_last, gamma)
    
    return out
