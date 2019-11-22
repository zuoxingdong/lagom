import numpy as np
import torch


def tensorify(x, device):
    if torch.is_tensor(x):
        if str(x.device) != str(device):
            x = x.to(device)
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    else:
        return torch.from_numpy(np.asarray(x)).float().to(device)
    
    
def numpify(x, dtype=None):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x
