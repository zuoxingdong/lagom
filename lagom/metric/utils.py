import numpy as np
import torch


def _wrap_Vs(Vs):
    if torch.is_tensor(Vs):
        Vs = Vs.squeeze().detach().cpu().numpy()
    else:
        Vs = np.asarray(Vs).squeeze()  # numpy: remove all single axes
    if Vs.ndim == 0:
        Vs = np.expand_dims(Vs, 0)
    assert Vs.ndim == 1, f'expected ndim=1, got {Vs.ndim}'
    return Vs


def _wrap_last_V(last_V):
    if torch.is_tensor(last_V):
        last_V = last_V.item()
    else:
        last_V = np.asarray(last_V).item()
    assert np.isscalar(last_V)
    return last_V
