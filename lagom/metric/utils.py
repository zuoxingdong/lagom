import numpy as np
import torch


def _wrap_Vs(Vs):
    if torch.is_tensor(Vs):
        Vs = Vs.squeeze(0).detach().cpu().numpy()
    else:
        Vs = np.asarray(Vs).squeeze()  # numpy: remove all single axes
    assert Vs.ndim == 1
    return Vs


def _wrap_last_V(last_V):
    if torch.is_tensor(last_V):
        last_V = last_V.item()
    else:
        last_V = np.asarray(last_V).item()
    assert np.isscalar(last_V)
    return last_V
