import numpy as np
import torch

from lagom.transform import ExpFactorCumSum

from .td import td0_error
from .td import _split_reshape


def gae(rewards, Vs, last_V, done, gamma, lam):
    td0 = td0_error(rewards, Vs, last_V, done, gamma)
    f = ExpFactorCumSum(gamma*lam)
    return f(td0).tolist()[0]


def get_gae(D, Vs, last_Vs, gamma, lam):
    r"""Calculate the Generalized Advantage Estimation (GAE) of a batch of episodic transitions.
    
    Let :math:`\delta_t` be the TD(0) error at time step :math:`t`, the GAE at time step :math:`t` is calculated
    as follows
    
    .. math::
        A_t^{\mathrm{GAE}(\gamma, \lambda)} = \sum_{k=0}^{\infty}(\gamma\lambda)^k \delta_{t + k}
    
    """
    if torch.is_tensor(Vs):
        assert Vs.ndimension() == 2
        Vs = Vs.detach().cpu().numpy().tolist()
    Vs = _split_reshape(Vs, D.Ts)
    if torch.is_tensor(last_Vs):
        assert last_Vs.ndimension() == 1
        last_Vs = last_Vs.detach().cpu().numpy().tolist()
    out = np.zeros((D.N, D.T), dtype=np.float32)
    for n in range(D.N):
        y = []
        for m in range(len(D.r[n])): 
            y += gae(D.r[n][m], Vs[n][m], last_Vs.pop(0), D.done[n][m][-1], gamma, lam)
        out[n, :len(y)] = y
    return out
