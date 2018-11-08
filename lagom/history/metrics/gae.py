from lagom.history import Segment

from lagom.transform import ExpFactorCumSum

from .td import td0_error_from_trajectory


def gae(list_td0_error, gamma, lam):
    r"""Calculate a list of generalized advantage estimation (GAE) given a list of TD(0) errors and discounted factor. 
    
    Let :math:`\delta_t` be the TD(0) error at time step :math:`t`, the GAE at time step :math:`t` is calculated
    as follows
    
    .. math::
        A_t^{\mathrm{GAE}(\gamma, \lambda)} = \sum_{k=0}^{\infty}(\gamma\lambda)^k \delta_{t + k}
    
    Args:
        list_td0_error (list): a list of TD(0) errors. 
        gamma (float): a discounted factor. 
        lam (float): GAE parameter. 
        
    Returns
    -------
    list_gae : list
        a list of GAE values.
    """
    assert isinstance(list_td0_error, list)
    assert gamma >= 0.0 and gamma <= 1.0
    assert lam >= 0.0 and lam <= 1.0
    
    out = ExpFactorCumSum(gamma*lam)(list_td0_error)
    
    return out


def gae_from_trajectory(trajectory, Vs, V_last, gamma, lam):
    r"""Return a list of GAE values from a trajectory. 
    
    Args:
        trajectory (Trajectory): a trajectory
        Vs (list): a list of state values for each time step excluding the final state. 
        V_last (object): the value of the final state in the trajectory
        gamma (float): discounted factor
        lam (float): GAE parameter
        
    Returns
    -------
    out : list
         a list of GAE values
    """
    list_td0_error = td0_error_from_trajectory(trajectory, Vs, V_last, gamma)
    
    out = gae(list_td0_error, gamma, lam)
    
    return out


def gae_from_segment(segment, all_Vs, all_V_last, gamma, lam):
    r"""Return a list of GAE values from a segment. 
    
    Args:
        segment (Segment): a segment
        all_Vs (list): a list of state values for each time step and each inner trajectory
            excluding the final state. 
        all_V_last (list): a list of final state values for each inner trajectory. 
        gamma (float): discounted factor
        lam (float): GAE parameter
        
    Returns
    -------
    out : list
         a list of GAE values
    """
    assert isinstance(segment, Segment)
    assert isinstance(all_Vs, list) and len(all_Vs) == len(segment.trajectories)
    assert len(all_V_last) == len(segment.trajectories)
    
    out = []
    for trajectory, Vs, V_last in zip(segment.trajectories, all_Vs, all_V_last):
        out += gae_from_trajectory(trajectory, Vs, V_last, gamma, lam)
        
    return out
