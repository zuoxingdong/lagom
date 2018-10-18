from lagom.networks import BaseNetwork

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Independent
from torch.distributions import Normal

from lagom.networks import ortho_init


class DiagGaussianHead(BaseNetwork):
    r"""Defines a diagonal Gaussian neural network head for continuous action space. 
    
    The network outputs the mean :math:`\mu` and log-variance :math:`\log\sigma^2` (optimize 
    in log-space i.e. negative, zero and positive. )
    
    There are several options for modelling the standard deviation:
    
    * :attr:`min_std` constrains the standard deviation with a lower bound threshould. This helps to avoid
      numerical instability, e.g. producing ``NaN``
        
    * :attr:`std_style` indicates the parameterization of the standard deviation. 

        * For std_style='exp', the standard deviation is obtained by applying exponentiation on log-variance
          i.e. :math:`\exp(0.5\log\sigma^2)`.
        * For std_style='softplus', the standard deviation is obtained by applying softplus operation on
          log-variance, i.e. :math:`f(x) = \log(1 + \exp(x))`.
            
    * :attr:`constant_std` indicates whether to use constant standard deviation or learning it instead.
      If a ``None`` is given, then the standard deviation will be learned. Note that a scalar value
      should be given if using constant value for all dimensions.
        
    * :attr:`std_state_dependent` indicates whether to learn standard deviation with dependency on state.
    
        * For std_state_dependent=``True``, the log-variance head is created and its forward pass takes
          last feature values as input. 
        * For std_state_dependent=``False``, the independent trainable nn.Parameter will be created. It
          does not need forward pass, but the backpropagation will calculate its gradients. 
            
    * :attr:`init_std` controls the initial values for independently learnable standard deviation. 
      Note that this is only valid when ``std_state_dependent=False``. 
    
    Example:
    
        >>> from lagom.envs import make_gym_env, EnvSpec
        >>> env = make_gym_env('Pendulum-v0', 0)
        >>> env_spec = EnvSpec(env)

        >>> head = DiagGaussianHead(None, None, 30, env_spec)
        >>> d = head(torch.randn(3, 30))
        >>> d.base_dist
        Normal(loc: torch.Size([3, 1]), scale: torch.Size([3, 1]))
    
    """
    def __init__(self, 
                 config, 
                 device, 
                 feature_dim, 
                 env_spec, 
                 min_std=1e-6, 
                 std_style='exp', 
                 constant_std=None,
                 std_state_dependent=False,
                 init_std=1.0,
                 **kwargs):
        self.feature_dim = feature_dim
        self.env_spec = env_spec
        assert self.env_spec.control_type == 'Continuous', 'expected as Continuous control type'
        
        self.min_std = min_std
        self.std_style = std_style
        self.constant_std = constant_std
        self.std_state_dependent = std_state_dependent
        self.init_std = init_std
        if self.constant_std is not None:
            assert not self.std_state_dependent
        
        super().__init__(config, device, **kwargs)
        
    def make_params(self, config):
        self.mean_head = nn.Linear(in_features=self.feature_dim, 
                                   out_features=self.env_spec.action_space.flat_dim)
        
        if self.constant_std is not None:
            if np.isscalar(self.constant_std):
                self.logvar_head = torch.full(size=[self.env_spec.action_space.flat_dim], 
                                              fill_value=torch.log(torch.tensor(self.constant_std)**2),
                                              requires_grad=False)
            else:
                self.logvar_head = torch.log(torch.from_numpy(np.array(self.constant_std)**2).float())
        else:  # learn it
            if self.std_state_dependent:
                self.logvar_head = nn.Linear(in_features=self.feature_dim, 
                                             out_features=self.env_spec.action_space.flat_dim)
            else:
                msg = f'expected init_std is given as scalar value, got {self.init_std}'
                assert self.init_std is not None, msg
                self.logvar_head = nn.Parameter(torch.full(size=[self.env_spec.action_space.flat_dim], 
                                                           fill_value=torch.log(torch.tensor(self.init_std)**2), 
                                                           requires_grad=True))
        self.logvar_head = self.logvar_head.to(self.device)
        
    def init_params(self, config):
        # 0.01->almost zeros initially
        ortho_init(self.mean_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)
        
        if isinstance(self.logvar_head, nn.Linear):
            # 0.01->almost 1.0 std
            ortho_init(self.logvar_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)
        
    def reset(self, cofnig, **kwargs):
        pass
    
    def forward(self, x):
        mean = self.mean_head(x)
        
        if isinstance(self.logvar_head, nn.Linear):
            logvar = self.logvar_head(x)
        else:
            logvar = self.logvar_head.expand_as(mean)
            
        if self.std_style == 'exp':
            std = torch.exp(0.5*logvar)
        elif self.std_style == 'softplus':
            std = F.softplus(logvar)
            
        min_std = torch.full(std.size(), self.min_std).type_as(std).to(self.device)
        std = torch.max(std, min_std)
        
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        
        return action_dist
