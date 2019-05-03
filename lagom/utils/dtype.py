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
    
    
def numpify(x, dtype):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(dtype)
    elif isinstance(x, np.ndarray):
        return x.astype(dtype)
    else:
        return np.asarray(x, dtype=dtype)
