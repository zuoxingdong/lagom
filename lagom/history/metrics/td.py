import numpy as np
import torch

from lagom.history import BatchEpisode
from lagom.history import BatchSegment


def td0_target_from_episode(batch_episode, all_Vs, last_Vs, gamma):
    r"""Calculate TD(0) targets of a batch of episodic transitions. 
    
    Let :math:`r_1, r_2, \dots, r_T` be a list of rewards and let :math:`V(s_0), V(s_1), \dots, V(s_{T-1}), V(s_{T})`
    be a list of state values including a last state value. Let :math:`\gamma` be a discounted factor, 
    the TD(0) targets are calculated as follows
        
    .. math::
        r_t + \gamma V(s_t), \forall t = 1, 2, \dots, T
        
    .. note::

        The state values for terminal states are masked out as zero !
    
    Args:
        batch_episode (BatchEpisode): a batch of episodic transitions. 
        all_Vs (object): the value of states in all transitions. 
        last_Vs (object): the value of the final states in the episode.
        gamma (float): discounted factor.
    
    Returns
    -------
    out : object
        TD(0) targets
    """
    assert isinstance(batch_episode, BatchEpisode)
    
    Vs = np.zeros((batch_episode.N, batch_episode.maxT+1), dtype=np.float32)
    for t, V in enumerate(all_Vs):
        if torch.is_tensor(V):
            Vs[:, t] = V.detach().cpu().numpy().squeeze(-1)
        else:
            Vs[:, t] = V
            
    if torch.is_tensor(last_Vs):
        last_Vs = last_Vs.detach().cpu().numpy().squeeze(-1)
    Vs[range(batch_episode.N), batch_episode.Ts] = last_Vs
    
    out = batch_episode.numpy_rewards + gamma*Vs[:, 1:]*batch_episode.numpy_masks
    
    return out
        
    
def td0_target_from_segment(batch_segment, all_Vs, last_Vs, gamma):
    r"""Calculate TD(0) targets of a batch of rolling segments. 
    
    Let :math:`r_1, r_2, \dots, r_T` be a list of rewards and let :math:`V(s_0), V(s_1), \dots, V(s_{T-1}), V(s_{T})`
    be a list of state values including a last state value. Let :math:`\gamma` be a discounted factor, 
    the TD(0) targets are calculated as follows
        
    .. math::
        r_t + \gamma V(s_t), \forall t = 1, 2, \dots, T
        
    .. note::

        The state values for terminal states are masked out as zero !
    
    Args:
        batch_segment (BatchSegment): a batch of rolling segments. 
        all_Vs (object): the value of states in all transitions. 
        last_Vs (object): the value of the final states in the episode.
        gamma (float): discounted factor.
    
    Returns
    -------
    out : object
        TD(0) targets
    """
    assert isinstance(batch_segment, BatchSegment)
    
    Vs = np.zeros((batch_segment.N, batch_segment.T+1), dtype=np.float32)
    for t, V in enumerate(all_Vs):
        if torch.is_tensor(V):
            Vs[:, t] = V.detach().cpu().numpy().squeeze(-1)
        else:
            Vs[:, t] = V
            
    if torch.is_tensor(last_Vs):
        last_Vs = last_Vs.detach().cpu().numpy().squeeze(-1)
    Vs[:, -1] = last_Vs
    
    out = batch_segment.numpy_rewards + gamma*Vs[:, 1:]*batch_segment.numpy_masks
    
    return out


def td0_error_from_episode(batch_episode, all_Vs, last_Vs, gamma):
    r"""Calculate TD(0) errors of a batch of episodic transitions. 
    
    Let :math:`r_1, r_2, \dots, r_T` be a list of rewards and let :math:`V(s_0), V(s_1), \dots, V(s_{T-1}), V(s_{T})`
    be a list of state values including a last state value. Let :math:`\gamma` be a discounted factor, 
    the TD(0) errors are calculated as follows
    
    .. math::
        \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
        
    .. note::

        The state values for terminal states are masked out as zero !
    
    Args:
        batch_episode (BatchEpisode): a batch of episodic transitions. 
        all_Vs (object): the value of states in all transitions. 
        last_Vs (object): the value of the final states in the episode.
        gamma (float): discounted factor.
    
    Returns
    -------
    out : object
        TD(0) errors
    """
    assert isinstance(batch_episode, BatchEpisode)

    Vs = np.zeros((batch_episode.N, batch_episode.maxT+1), dtype=np.float32)
    for t, V in enumerate(all_Vs):
        if torch.is_tensor(V):
            Vs[:, t] = V.detach().cpu().numpy().squeeze(-1)
        else:
            Vs[:, t] = V

    if torch.is_tensor(last_Vs):
        last_Vs = last_Vs.detach().cpu().numpy().squeeze(-1)
    Vs[range(batch_episode.N), batch_episode.Ts] = last_Vs
            
    out = batch_episode.numpy_rewards + gamma*Vs[:, 1:]*batch_episode.numpy_masks
    for n, T in enumerate(batch_episode.Ts):
        Vs[n, T:] = 0.0
    out = out - Vs[:, :-1]

    return out


def td0_error_from_segment(batch_segment, all_Vs, last_Vs, gamma):
    r"""Calculate TD(0) errors of a batch of rolling segments. 
    
    Let :math:`r_1, r_2, \dots, r_T` be a list of rewards and let :math:`V(s_0), V(s_1), \dots, V(s_{T-1}), V(s_{T})`
    be a list of state values including a last state value. Let :math:`\gamma` be a discounted factor, 
    the TD(0) errors are calculated as follows
    
    .. math::
        \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
        
    .. note::

        The state values for terminal states are masked out as zero !
    
    Args:
        batch_segment (BatchSegment): a batch of rolling segments. 
        all_Vs (object): the value of states in all transitions. 
        last_Vs (object): the value of the final states in the episode.
        gamma (float): discounted factor.
    
    Returns
    -------
    out : object
        TD(0) errors
    """
    assert isinstance(batch_segment, BatchSegment)
    
    Vs = np.zeros((batch_segment.N, batch_segment.T+1), dtype=np.float32)
    for t, V in enumerate(all_Vs):
        if torch.is_tensor(V):
            Vs[:, t] = V.detach().cpu().numpy().squeeze(-1)
        else:
            Vs[:, t] = V
            
    if torch.is_tensor(last_Vs):
        last_Vs = last_Vs.detach().cpu().numpy().squeeze(-1)
    Vs[:, -1] = last_Vs
    
    print(Vs)
    
    out = batch_segment.numpy_rewards + gamma*Vs[:, 1:]*batch_segment.numpy_masks - Vs[:, :-1]
    
    return out
