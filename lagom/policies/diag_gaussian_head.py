from lagom.networks import BaseNetwork

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Independent
from torch.distributions import Normal

from lagom.networks import ortho_init


class DiagGaussianHead(BaseNetwork):
    r"""Defines a diagonal Gaussian neural network head for continuous action space. 
    
    The network outputs the mean :math:`\mu` and logarithm of standard deviation :math:`\log\sigma` (optimize 
    in log-space i.e. negative, zero and positive. )
    
    There are several options for modelling the standard deviation:
    
    * :attr:`min_std` constrains the standard deviation with a lower bound threshould. This helps to avoid
      numerical instability, e.g. producing ``NaN``
        
    * :attr:`std_style` indicates the parameterization of the standard deviation. 

        * For std_style='exp', the standard deviation is obtained by applying exponentiation on log standard
          deviation i.e. :math:`\exp(\log\sigma)`.
        * For std_style='softplus', the standard deviation is modeled by softplus, i.e. :math:`\log(1 + \exp(x))`.
        * For std_style='sigmoidal', the standard deviation is modeled by :math:`0.01 + 2\mathrm{sigmoid}(x)`.
            
    * :attr:`constant_std` indicates whether to use constant standard deviation or learning it instead.
      If a ``None`` is given, then the standard deviation will be learned. Note that a scalar value
      should be given if using constant value for all dimensions.
        
    * :attr:`std_state_dependent` indicates whether to learn standard deviation with dependency on state.
    
        * For std_state_dependent=``True``, the log-std head is created and its forward pass takes
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
                 init_std=0.6,
                 **kwargs):
        self.feature_dim = feature_dim
        self.env_spec = env_spec
        assert self.env_spec.control_type == 'Continuous', 'expected as Continuous control type'
        
        assert std_style in ['exp', 'softplus', 'sigmoidal']
        self.std_style = std_style
        
        self.min_std = min_std
        
        if constant_std is None:
            self.logit_constant_std = None
        else:
            self.logit_constant_std = self._compute_logit(constant_std, self.std_style)
        self.std_state_dependent = std_state_dependent
        
        self.logit_init_std = self._compute_logit(init_std, self.std_style)
        
        super().__init__(config, device, **kwargs)
        
    def make_params(self, config):
        self.mean_head = nn.Linear(in_features=self.feature_dim, 
                                   out_features=self.env_spec.action_space.flat_dim)
        
        if self.logit_constant_std is not None:
            self.logstd_head = torch.full([self.env_spec.action_space.flat_dim], 
                                          self.logit_constant_std,
                                          requires_grad=False)
        else:  # learn it
            if self.std_state_dependent:
                self.logstd_head = nn.Linear(in_features=self.feature_dim, 
                                             out_features=self.env_spec.action_space.flat_dim)
            else:
                self.logstd_head = nn.Parameter(torch.full([self.env_spec.action_space.flat_dim], 
                                                           self.logit_init_std, 
                                                           requires_grad=True))
        
    def init_params(self, config):
        # 0.01->almost zeros initially
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        
        if isinstance(self.logstd_head, nn.Linear):
            # 0.01->small std
            ortho_init(self.logstd_head, weight_scale=0.01, constant_bias=0.0)
        
    def reset(self, config, **kwargs):
        pass
    
    def forward(self, x):
        mean = self.mean_head(x)
        
        if isinstance(self.logstd_head, nn.Linear):
            logstd = self.logstd_head(x)
        else:
            logstd = self.logstd_head.expand_as(mean)
            
        if self.std_style == 'exp':
            std = torch.exp(logstd)
        elif self.std_style == 'softplus':
            std = F.softplus(logstd)
        elif self.std_style == 'sigmoidal':
            std = 0.01 + 2*torch.sigmoid(logstd)
            
        std = std.clamp_min(self.min_std)

        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        
        return action_dist
    
    def _compute_logit(self, x, std_style):
        if std_style == 'exp':
            return math.log(x)
        elif std_style == 'softplus':
            return math.log(math.exp(x) - 1)
        elif std_style == 'sigmoidal':
            return -math.log(2/(x - 0.01) - 1)
