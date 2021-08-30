import torch
import torch.nn.functional as F


def bound_logvar(logvar, min_var, max_var):
    min_logvar = torch.as_tensor(min_var).float().log()
    delta_logvar = torch.as_tensor(max_var - min_var).float().log()
    logvar = delta_logvar - F.softplus(-logvar + delta_logvar)
    logvar = min_logvar + F.softplus(logvar - min_logvar)
    return logvar
