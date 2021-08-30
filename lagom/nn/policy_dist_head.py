import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Independent
from torch.distributions import Normal

from .module import Module
from .init import ortho_init
from .bound_logvar import bound_logvar


class CategoricalHead(Module):
    r"""Defines a module for a Categorical (discrete) action distribution. 
    
    Example:
    
        >>> import torch
        >>> action_head = CategoricalHead(30, 4, 'cpu')
        >>> action_head(torch.randn(2, 30))
        Categorical(probs: torch.Size([2, 4]))
        
    Args:
        feature_dim (int): number of input features
        num_action (int): number of discrete actions
        **kwargs: keyword arguments for more specifications.
    
    """
    def __init__(self, feature_dim, num_action, **kwargs):
        super().__init__(**kwargs)
        
        self.feature_dim = feature_dim
        self.num_action = num_action
        
        self.action_head = nn.Linear(self.feature_dim, self.num_action)
        # weight_scale=0.01 -> uniformly distributed
        ortho_init(self.action_head, weight_scale=0.01, constant_bias=0.0)

    def forward(self, x):
        action_score = self.action_head(x)
        action_prob = F.softmax(action_score, dim=-1)
        action_dist = Categorical(probs=action_prob)
        return action_dist


class DiagGaussianHead(Module):
    r"""Defines a module for a diagonal Gaussian (continuous) action distribution.
    
    The standard deviation can be either state-independent or state-dependent. 
    
    The network outputs the mean :math:`\mu(x)` and the logarithm of variance :math:`\log\sigma^2` 
    which allows to optimize in log-space, i.e. both negative and positive. 
    
    The standard deviation is obtained by applying exponential function :math:`\exp(0.5\log\sigma^2)`.
    
    The log-variance can also be bounded above and below. 
    
    Example:
    
        >>> import torch
        >>> action_head = DiagGaussianHead(10, 4, 0.45)
        >>> action_dist = action_head(torch.randn(2, 10))
        >>> action_dist.base_dist
        Normal(loc: torch.Size([2, 4]), scale: torch.Size([2, 4]))
        >>> action_dist.base_dist.stddev
        tensor([[0.4500, 0.4500, 0.4500, 0.4500],
                [0.4500, 0.4500, 0.4500, 0.4500]], grad_fn=<ExpBackward>)
    
    Args:
        feature_dim (int): number of input features
        action_dim (int): flat dimension of actions
        std0 (float): initial standard deviation
        **kwargs: keyword arguments for more specifications.
    
    """
    def __init__(self, feature_dim, action_dim, std_mode='independent', std0=None, min_var=None, max_var=None, **kwargs):
        super().__init__(**kwargs)
        assert std_mode in ['independent', 'dependent']
        if std_mode == 'independent':
            assert std0 is not None and std0 > 0
        elif std_mode == 'dependent':
            assert std0 is None
        if min_var is None:
            assert max_var is None
        else:
            assert max_var is not None
            assert min_var < max_var
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.std_mode = std_mode
        self.std0 = std0
        self.min_var = min_var
        self.max_var = max_var
        
        self.mean_head = nn.Linear(feature_dim, action_dim)
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)  # 0.01 -> almost zeros initially
        if std_mode == 'independent':
            self.logvar_head = nn.Parameter(torch.full((action_dim,), math.log(std0**2)))
        elif std_mode == 'dependent':
            self.logvar_head = nn.Linear(feature_dim, action_dim)
            ortho_init(self.logvar_head, weight_scale=0.01, constant_bias=0.0)

    def forward(self, x):
        mean = self.mean_head(x)
        if self.std_mode == 'independent':
            logvar = self.logvar_head.expand_as(mean)
        elif self.std_mode == 'dependent':
            logvar = self.logvar_head(x)
        if self.min_var is not None and self.max_var is not None:
            logvar = bound_logvar(logvar, self.min_var, self.max_var)
        std = torch.exp(0.5*logvar)
        dist = Independent(Normal(loc=mean, scale=std), 1)
        return dist
