import numpy as np
import torch


def _split_reshape(Vs, Ts):
    out = []
    for n, T in enumerate(Ts):
        y = []
        for t in T:
            y.append([Vs[n].pop(0) for _ in range(t)])
        out.append(y)
    return out


def td0_target(rewards, Vs, last_V, done, gamma):
    rewards = np.asarray(rewards)
    if done:
        all_Vs = np.append(Vs, 0.0)
    else:
        all_Vs = np.append(Vs, last_V)
    out = rewards + gamma*all_Vs[1:]
    return out.tolist()


def get_td0_target(D, Vs, last_Vs, gamma):
    r"""Calculate TD(0) targets of a batch of episodic transitions. 
    
    Let :math:`r_1, r_2, \dots, r_T` be a list of rewards and let :math:`V(s_0), V(s_1), \dots, V(s_{T-1}), V(s_{T})`
    be a list of state values including a last state value. Let :math:`\gamma` be a discounted factor, 
    the TD(0) targets are calculated as follows
        
    .. math::
        r_t + \gamma V(s_t), \forall t = 1, 2, \dots, T
        
    .. note::

        The state values for terminal states are masked out as zero !
    
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
            y += td0_target(D.r[n][m], Vs[n][m], last_Vs.pop(0), D.done[n][m][-1], gamma)
        out[n, :len(y)] = y
    return out
            

def td0_error(rewards, Vs, last_V, done, gamma):
    rewards = np.asarray(rewards)
    if done:
        all_Vs = np.append(Vs, 0.0)
    else:
        all_Vs = np.append(Vs, last_V)
    out = rewards + gamma*all_Vs[1:] - all_Vs[:-1]
    return out.tolist()


def get_td0_error(D, Vs, last_Vs, gamma):
    r"""Calculate TD(0) errors of a batch of episodic transitions. 
    
    Let :math:`r_1, r_2, \dots, r_T` be a list of rewards and let :math:`V(s_0), V(s_1), \dots, V(s_{T-1}), V(s_{T})`
    be a list of state values including a last state value. Let :math:`\gamma` be a discounted factor, 
    the TD(0) errors are calculated as follows
    
    .. math::
        \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
        
    .. note::

        The state values for terminal states are masked out as zero !
    
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
            y += td0_error(D.r[n][m], Vs[n][m], last_Vs.pop(0), D.done[n][m][-1], gamma)
        out[n, :len(y)] = y
    return out
