import numpy as np
import torch

from lagom.transform import geometric_cumsum


def bootstrapped_returns(rewards, last_V, done, gamma):
    r"""Return (discounted) accumulated returns with bootstrapping for a 
    batch of episodic transitions. 
    
    Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
    .. math::
        Q_t = r_t + \gamma r_{t+1} + \dots + \gamma^{T - t} r_T + \gamma^{T - t + 1} V(s_{T+1})
        
    .. note::

        The state values for terminal states are masked out as zero !

    """
    if done:
        out = geometric_cumsum(gamma, rewards + [0.0])
    else:
        out = geometric_cumsum(gamma, rewards + [last_V])
    return out[0, :-1].tolist()


def get_bootstrapped_returns(D, last_Vs, gamma):
    if torch.is_tensor(last_Vs):
        assert last_Vs.ndimension() == 1
        last_Vs = last_Vs.detach().cpu().numpy().tolist()
    out = np.zeros((D.N, D.T), dtype=np.float32)
    for n in range(D.N):
        y = []
        for m in range(len(D.r[n])):
            y += bootstrapped_returns(D.r[n][m], last_Vs.pop(0), D.done[n][m][-1], gamma)
        out[n, :len(y)] = y
    return out
