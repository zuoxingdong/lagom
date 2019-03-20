import numpy as np

from lagom.transform import geometric_cumsum

from .utils import _wrap_last_V


def returns(gamma, traj):
    return geometric_cumsum(gamma, traj.rewards)[0].astype(np.float32)


def bootstrapped_returns(gamma, traj, last_V):
    r"""Return (discounted) accumulated returns with bootstrapping for a 
    batch of episodic transitions. 
    
    Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
    .. math::
        Q_t = r_t + \gamma r_{t+1} + \dots + \gamma^{T - t} r_T + \gamma^{T - t + 1} V(s_{T+1})
        
    .. note::

        The state values for terminal states are masked out as zero !

    """
    last_V = _wrap_last_V(last_V)
    
    if traj.completed:
        out = geometric_cumsum(gamma, traj.rewards + [0.0])
    else:
        out = geometric_cumsum(gamma, traj.rewards + [last_V])
    return out[0, :-1].astype(np.float32)
