import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent
from torch.distributions import Normal

from .module import Module
from .init import ortho_init


class DiagGaussianHead(Module):
    r"""Defines a module for a diagonal Gaussian (continuous) action distribution which
    the standard deviation is state independent. 
    
    The network outputs the mean :math:`\mu(x)` and the state independent logarithm of variance 
    :math:`\log\sigma^2` (allowing to optimize in log-space, i.e. both negative and positive). 
    
    The variance is obtained by applying exponential function :math:`\epsilon + \exp(x)` 
    where :math:`\epsilon=1e-4` is a lower bound to avoid numerical instability, e.g. producing ``NaN``. 
    Then the standard deviation is obtained by taking the square root. 
    
    Example:
    
        >>> import torch
        >>> action_head = DiagGaussianHead(10, 4, 'cpu', 0.45)
        >>> action_dist = action_head(torch.randn(2, 10))
        >>> action_dist.base_dist
        Normal(loc: torch.Size([2, 4]), scale: torch.Size([2, 4]))
        >>> action_dist.base_dist.stddev
        tensor([[0.4500, 0.4500, 0.4500, 0.4500],
                [0.4500, 0.4500, 0.4500, 0.4500]], grad_fn=<SqrtBackward>)
    
    Args:
        feature_dim (int): number of input features
        action_dim (int): flat dimension of actions
        device (torch.device): PyTorch device
        std0 (float): initial standard deviation
        **kwargs: keyword arguments for more specifications.
    
    """
    def __init__(self, feature_dim, action_dim, device, std0, **kwargs):
        super().__init__(**kwargs)
        self.eps = 1e-4  # used for default min-variance to avoid numerical instability
        assert std0 > 0
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.device = device
        self.std0 = std0
        
        self.logvar0 = math.log(std0**2 - self.eps)
        
        self.mean_head = nn.Linear(self.feature_dim, self.action_dim)
        # 0.01 -> almost zeros initially
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        self.logvar_head = nn.Parameter(torch.full((self.action_dim,), self.logvar0, requires_grad=True))
        
        self.to(self.device)
        
    def forward(self, x):
        mean = self.mean_head(x)
        logvar = self.logvar_head.expand_as(mean)
        var = self.eps + torch.exp(logvar)
        std = torch.sqrt(var)
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        return action_dist
