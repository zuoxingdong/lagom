import numpy as np
import torch

from lagom.history import BatchEpisode
from lagom.history import BatchSegment

from lagom.transform import ExpFactorCumSum


def bootstrapped_returns_from_episode(batch_episode, last_Vs, gamma):
    r"""Return (discounted) accumulated returns with bootstrapping for a 
    batch of episodic transitions. 
    
    Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
    .. math::
        Q_t = r_t + \gamma r_{t+1} + \dots + \gamma^{T - t} r_T + \gamma^{T - t + 1} V(s_{T+1})
        
    .. note::

        The state values for terminal states are masked out as zero !
    
    Args:
        batch_episode (BatchEpisode): a batch of episodic transitions. 
        last_Vs (object): the value of the final states in the episode.
        gamma (float): discounted factor
        
    Returns
    -------
    out : ndarray
        an array of (discounted) bootstrapped returns.
    """
    assert isinstance(batch_episode, BatchEpisode)
    
    if torch.is_tensor(last_Vs):
        last_Vs = last_Vs.detach().cpu().numpy().squeeze(-1)
    last_Vs = last_Vs*np.logical_not(batch_episode.completes).astype(np.float32)
    
    f = ExpFactorCumSum(gamma)
    out = np.concatenate([batch_episode.numpy_rewards, np.zeros((batch_episode.N, 1), dtype=np.float32)], axis=1)
    out[range(batch_episode.N), batch_episode.Ts] = last_Vs
    out = f(out)
    out[range(batch_episode.N), batch_episode.Ts] = 0.0
    out = out[:, :-1]
    
    return out
    
    
def bootstrapped_returns_from_segment(batch_segment, last_Vs, gamma):
    r"""Return (discounted) accumulated returns with bootstrapping for a 
    batch of rolling segment. 
    
    Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
    .. math::
        Q_t = r_t + \gamma r_{t+1} + \dots + \gamma^{T - t} r_T + \gamma^{T - t + 1} V(s_{T+1})
        
    .. note::

        The state values for terminal states are masked out as zero !
    
    Args:
        batch_segment (BatchSegment): a batch of rolling segments. 
        last_Vs (object): the value of the final states in the episode.
        gamma (float): discounted factor
        
    Returns
    -------
    out : ndarray
        an array of (discounted) bootstrapped returns.
    """
    assert isinstance(batch_segment, BatchSegment)
    
    if torch.is_tensor(last_Vs):
        last_Vs = last_Vs.detach().cpu().numpy().squeeze(-1)
    
    f = ExpFactorCumSum(gamma)
    mask = np.concatenate([batch_segment.numpy_masks, np.ones((batch_segment.N, 1), dtype=np.float32)], axis=1)
    out = np.zeros((batch_segment.N, batch_segment.T+1), dtype=np.float32)
    out[:, :-1] = batch_segment.numpy_rewards
    out[:, -1] = last_Vs
    out = f(out, mask=mask)[:, :-1]
    
    return out
