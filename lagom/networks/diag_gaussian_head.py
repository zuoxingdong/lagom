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
    
    The network outputs the mean :math:`\mu(x)` and the state independent logarithm of standard 
    deviation :math:`\log\sigma` (allowing to optimize in log-space, i.e. both negative and positive). 
    
    The standard deviation is obtained by applying exponential function :math:`\exp(x)`.
    
    Example:
    
        >>> import torch
        >>> action_head = DiagGaussianHead(10, 4, 'cpu', 0.45)
        >>> action_dist = action_head(torch.randn(2, 10))
        >>> action_dist.base_dist
        Normal(loc: torch.Size([2, 4]), scale: torch.Size([2, 4]))
        >>> action_dist.base_dist.stddev
        tensor([[0.4500, 0.4500, 0.4500, 0.4500],
                [0.4500, 0.4500, 0.4500, 0.4500]], grad_fn=<ExpBackward>)
    
    Args:
        feature_dim (int): number of input features
        action_dim (int): flat dimension of actions
        device (torch.device): PyTorch device
        std0 (float): initial standard deviation
        **kwargs: keyword arguments for more specifications.
    
    """
    def __init__(self, feature_dim, action_dim, device, std0, **kwargs):
        super().__init__(**kwargs)
        assert std0 > 0
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.device = device
        self.std0 = std0
        
        self.mean_head = nn.Linear(self.feature_dim, self.action_dim)
        # 0.01 -> almost zeros initially
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        self.logstd_head = nn.Parameter(torch.full((self.action_dim,), math.log(std0)))
        
        self.to(self.device)
        
    def forward(self, x):
        mean = self.mean_head(x)
        logstd = self.logstd_head.expand_as(mean)
        std = torch.exp(logstd)
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        return action_dist
