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
    
    There are several options for modelling the variance term:
    
    * std_style='exp': regard raw features as log-variance, and variance is obtained by applying
      exponential function :math:`\epsilon + \exp(x)` where :math:`\epsilon=1e-4` is a lower bound to 
      avoid numerical instability, e.g. producing ``NaN``. 
    
    * std_style='softplus': the variance is obtained by applying softplus function
      :math:`\epsilon + \log(1 + \exp(x))` where :math:`\epsilon=1e-4` is a lower bound to 
      avoid numerical instability, e.g. producing ``NaN``. 
    
    * std_style='sigmoidal': the variance is obtained by applying sigmoidal function
      :math:`\sigma(x, \beta) = \frac{1}{1 + \beta\exp(-x)}` where :math:`\beta` is a
      scaling coefficient. And the function is bounded above and below by applying
      the transformation :math:`\min + (\max - \min)\sigma(x, \beta)`
    
    Then the standard deviation is obtained by taking the square root. 
    
    Example:
    
        >>> import torch
        >>> action_head = DiagGaussianHead(10, 4, 'cpu', 0.45, 'sigmoidal', [0.01, 1.0], 1.0)
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
        std_style (str): specifies the transformation mapping from raw features to variance:
            ['exp', 'softplus', 'sigmoidal']. Note that except for 'sigmoidal', both :attr:`std_range` 
            and :attr:`beta` should be ``None``.
        std_range (tuple/list, optional): lower and upper bound for sigmoid function
        beta (float, optional): scaling coefficient of sigmoid function
        **kwargs: keyword arguments for more specifications.
    
    """
    def __init__(self, 
                 feature_dim, 
                 action_dim, 
                 device, 
                 std0,
                 std_style, 
                 std_range=None, 
                 beta=None, 
                 **kwargs):
        super().__init__(**kwargs)
        
        assert std0 > 0
        assert std_style in ['exp', 'softplus', 'sigmoidal']
        if std_style == 'sigmoidal':
            assert len(std_range) == 2
            assert std_range[0] > 0 and std_range[1] > 0
            assert std_range[0] < std_range[1]
            assert std0 >= std_range[0] and std0 <= std_range[1]
            assert beta is not None and isinstance(beta, float)
        else:
            assert std_range is None, f'for std_style!=sigmoidal, expected None, got {std_range}'
            assert beta is None, f'for std_style!=sigmoidal, expected None, got {beta}'
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.device = device
        self.std0 = std0
        self.std_style = std_style
        self.std_range = std_range
        self.beta = beta
        
        self.eps = 1e-4  # used for default min-variance to avoid numerical instability
        
        self.mean_head = nn.Linear(self.feature_dim, self.action_dim)
        # 0.01 -> almost zeros initially
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        self.logvar_head = nn.Parameter(torch.full((self.action_dim,), 
                                                   self._get_logvar0(self.std0), 
                                                   requires_grad=True))
        
        self.to(self.device)
        
    def forward(self, x):
        mean = self.mean_head(x)
        logvar = self.logvar_head.expand_as(mean)
        var = self._get_var(logvar)
        std = torch.sqrt(var)
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        return action_dist
        
    def _get_logvar0(self, std0):
        var0 = std0**2
        if self.std_style == 'exp':
            return math.log(var0 - self.eps)
        elif self.std_style == 'softplus':
            return math.log(math.exp(var0 - self.eps) - 1)
        else:  # bounded beta-sigmoid
            min_std, max_std = self.std_range
            min_var = min_std**2
            max_var = max_std**2
            x = (var0 - min_var)/(max_var - min_var)
            x = float(np.clip(x, 1e-4, 0.9999))  # avoid too large +/- values
            x = -math.log((1/x - 1)/self.beta)
            return x
        
    def _get_var(self, logvar):
        if self.std_style == 'exp':
            return self.eps + torch.exp(logvar)
        elif self.std_style == 'softplus':
            return self.eps + F.softplus(logvar)
        else:  # bounded beta-sigmoid
            var = 1/(1 + self.beta*torch.exp(-logvar))
            min_std, max_std = self.std_range
            min_var = min_std**2
            max_var = max_std**2
            return min_var + (max_var - min_var)*var
