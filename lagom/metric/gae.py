from lagom.history import BatchEpisode
from lagom.history import BatchSegment

from lagom.transform import ExpFactorCumSum

from .td import td0_error_from_episode
from .td import td0_error_from_segment


def gae_from_episode(batch_episode, all_Vs, last_Vs, gamma, lam):
    r"""Calculate the Generalized Advantage Estimation (GAE) of a batch of episodic transitions.
    
    Let :math:`\delta_t` be the TD(0) error at time step :math:`t`, the GAE at time step :math:`t` is calculated
    as follows
    
    .. math::
        A_t^{\mathrm{GAE}(\gamma, \lambda)} = \sum_{k=0}^{\infty}(\gamma\lambda)^k \delta_{t + k}
    
    Args:
        batch_episode (BatchEpisode): a batch of episodic transitions. 
        all_Vs (object): the value of states in all transitions. 
        last_Vs (object): the value of the final states in the episode.
        gamma (float): discounted factor.
        lam (float): GAE parameter. 
        
    Returns
    -------
    out : object
        GAE values
    """
    assert isinstance(batch_episode, BatchEpisode)
    
    td0_error = td0_error_from_episode(batch_episode, all_Vs, last_Vs, gamma)
    f = ExpFactorCumSum(gamma*lam)
    out = f(td0_error)
    
    return out


def gae_from_segment(batch_segment, all_Vs, last_Vs, gamma, lam):
    r"""Calculate the Generalized Advantage Estimation (GAE) of a batch of rolling segments.
    
    Let :math:`\delta_t` be the TD(0) error at time step :math:`t`, the GAE at time step :math:`t` is calculated
    as follows
    
    .. math::
        A_t^{\mathrm{GAE}(\gamma, \lambda)} = \sum_{k=0}^{\infty}(\gamma\lambda)^k \delta_{t + k}
    
    Args:
        batch_segment (BatchSegment): a batch of rolling segments. 
        all_Vs (object): the value of states in all transitions. 
        last_Vs (object): the value of the final states in the episode.
        gamma (float): discounted factor.
        lam (float): GAE parameter. 
        
    Returns
    -------
    out : object
        GAE values
    """
    assert isinstance(batch_segment, BatchSegment)
    
    td0_error = td0_error_from_segment(batch_segment, all_Vs, last_Vs, gamma)
    f = ExpFactorCumSum(gamma*lam)
    out = f(td0_error, mask=batch_segment.numpy_masks)
    
    return out
