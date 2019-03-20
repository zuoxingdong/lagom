import numpy as np

from lagom.transform import geometric_cumsum

from .td import td0_error


def gae(gamma, lam, traj, Vs, last_V):
    r"""Calculate the Generalized Advantage Estimation (GAE) of a batch of episodic transitions.
    
    Let :math:`\delta_t` be the TD(0) error at time step :math:`t`, the GAE at time step :math:`t` is calculated
    as follows
    
    .. math::
        A_t^{\mathrm{GAE}(\gamma, \lambda)} = \sum_{k=0}^{\infty}(\gamma\lambda)^k \delta_{t + k}
    
    """
    delta = td0_error(gamma, traj, Vs, last_V)
    return geometric_cumsum(gamma*lam, delta)[0].astype(np.float32)
