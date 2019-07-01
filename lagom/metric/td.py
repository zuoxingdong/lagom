import numpy as np

from lagom.utils import numpify


def td0_target(gamma, traj, Vs, last_V):
    r"""Calculate TD(0) targets of a batch of episodic transitions. 
    
    Let :math:`r_1, r_2, \dots, r_T` be a list of rewards and let :math:`V(s_0), V(s_1), \dots, V(s_{T-1}), V(s_{T})`
    be a list of state values including a last state value. Let :math:`\gamma` be a discounted factor, 
    the TD(0) targets are calculated as follows
        
    .. math::
        r_t + \gamma V(s_t), \forall t = 1, 2, \dots, T
        
    .. note::

        The state values for terminal states are masked out as zero !
    
    """
    Vs = numpify(Vs, np.float32)
    last_V = numpify(last_V, np.float32)
    
    if traj.reach_terminal:
        Vs = np.append(Vs, 0.0)
    else:
        Vs = np.append(Vs, last_V)
    out = traj.numpy_rewards + gamma*Vs[1:]
    return out.astype(np.float32)


def td0_error(gamma, traj, Vs, last_V):
    r"""Calculate TD(0) errors of a batch of episodic transitions. 
    
    Let :math:`r_1, r_2, \dots, r_T` be a list of rewards and let :math:`V(s_0), V(s_1), \dots, V(s_{T-1}), V(s_{T})`
    be a list of state values including a last state value. Let :math:`\gamma` be a discounted factor, 
    the TD(0) errors are calculated as follows
    
    .. math::
        \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
        
    .. note::

        The state values for terminal states are masked out as zero !
    
    """
    Vs = numpify(Vs, np.float32)
    last_V = numpify(last_V, np.float32)
    
    if traj.reach_terminal:
        Vs = np.append(Vs, 0.0)
    else:
        Vs = np.append(Vs, last_V)
    out = traj.numpy_rewards + gamma*Vs[1:] - Vs[:-1]
    return out.astype(np.float32)
