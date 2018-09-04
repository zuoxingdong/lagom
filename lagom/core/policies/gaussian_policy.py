from .base_policy import BasePolicy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Independent
from torch.distributions import Normal

from lagom.core.networks import ortho_init


class GaussianPolicy(BasePolicy):
    r"""A parameterized policy defined as independent Gaussian distributions over a continuous action space. 
    
    .. note::
    
        The neural network given to the policy should define all but the final output layer. The final
        output layer for the Gaussian (Normal) distribution will be created with the policy and augmented
        to the network. This decoupled design makes it more flexible to use for different continuous
        action spaces. Note that the network must have an attribute ``.last_feature_dim`` of type
        ``int`` for the policy to create the final output layer (fully-connected), and this is
        recommended to be done in the method :meth:`~BaseNetwork.make_params` of the network class.
        The network outputs the mean :math:`\mu` and log-variance :math:`\log\sigma^2` which allows
        the network to optimize in log-space i.e. negative, zero and positive. 
        
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
    
        * For std_state_dependent=True, the log-variance head is created and its forward pass takes
          last feature values as input. 
        * For std_state_dependent=False, the independent trainable nn.Parameter will be created. It
          does not need forward pass, but the backpropagation will calculate its gradients. 
            
    * :attr:`init_std` controls the initial values for independently learnable standard deviation. 
      Note that this is only valid when :attr:`std_state_dependent`=False. 
    
    Example::
    
        >>> policy = GaussianPolicy(config=config, 
                                    network=network, 
                                    env_spec=env_spec, 
                                    min_std=1e-06, 
                                    std_style='exp', 
                                    constant_std=None, 
                                    std_state_dependent=True, 
                                    init_std=None)
        >>> policy(observation)
    """
    def __init__(self,
                 config,
                 network, 
                 env_spec,
                 min_std=1e-6, 
                 std_style='exp', 
                 constant_std=None,
                 std_state_dependent=True,
                 init_std=None,
                 **kwargs):
        
        super().__init__(config=config, network=network, env_spec=env_spec, **kwargs)
        
        # Record additional arguments
        self.min_std = min_std
        self.std_style = std_style
        self.constant_std = constant_std
        self.std_state_dependent = std_state_dependent
        self.init_std = init_std
        
        assert self.env_spec.control_type == 'Continuous', 'expected as Continuous control type'
        assert hasattr(self.network, 'last_feature_dim'), 'network expected to have an attribute `.last_feature_dim`'
        
        # Create mean head
        mean_head = nn.Linear(in_features=self.network.last_feature_dim, 
                              out_features=self.env_spec.action_space.flat_dim)
        # Create logvar head
        if self.constant_std is not None:  # using constant std
            if np.isscalar(self.constant_std):  # scalar
                logvar_head = torch.full(size=[self.env_spec.action_space.flat_dim], 
                                         fill_value=torch.log(torch.tensor(self.constant_std)**2),  # log(std**2)
                                         requires_grad=False)  # no grad
            else:  # a numpy array
                logvar_head = torch.log(torch.from_numpy(np.array(self.constant_std)**2).float())
        else:  # no constant std, so learn it
            if self.std_state_dependent:  # state dependent, so a layer
                logvar_head = nn.Linear(in_features=self.network.last_feature_dim, 
                                        out_features=self.env_spec.action_space.flat_dim)
            else:  # state independent, so a learnable nn.Parameter
                assert self.init_std is not None, f'expected init_std is given as scalar value, got {self.init_std}'
                logvar_head = nn.Parameter(torch.full(size=[self.env_spec.action_space.flat_dim], 
                                                      fill_value=torch.log(torch.tensor(self.init_std)**2), 
                                                      requires_grad=True))  # with grad
        
        # Orthogonal initialization to the parameters with scale 0.01, i.e. almost zero action
        ortho_init(mean_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)
        if isinstance(logvar_head, nn.Linear):  # linear layer with weights and bias
            ortho_init(logvar_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)
        
        # Augment to network (e.g. tracked by network.parameters() for optimizer to update)
        self.network.add_module('mean_head', mean_head)
        if isinstance(logvar_head, nn.Linear):
            self.network.add_module('logvar_head', logvar_head)
        else:
            self.network.logvar_head = logvar_head
    
    def __call__(self, x):
        # Output dictionary
        out_policy = {}
        
        # Forward pass through neural network for the input
        features = self.network(x)
        
        # Forward pass through mean head to obtain mean values for Gaussian distribution
        mean = self.network.mean_head(features)
        # Obtain logvar based on the options
        if isinstance(self.network.logvar_head, nn.Linear):  # linear layer, then do forward pass
            logvar = self.network.logvar_head(features)
        else:  # either Tensor or nn.Parameter
            logvar = self.network.logvar_head
            # Expand as same shape as mean
            logvar = logvar.expand_as(mean)
            # Same device with mean
            logvar = logvar.to(mean.device)
        
        # Get std from logvar
        if self.std_style == 'exp':
            std = torch.exp(0.5*logvar)
        elif self.std_style == 'softplus':
            std = F.softplus(logvar)
        
        # Lower bound threshould for std
        min_std = torch.full(std.size(), self.min_std).type_as(std).to(std.device)
        std = torch.max(std, min_std)
        
        # Create independent Gaussian distributions i.e. Diagonal Gaussian
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        # Sample action from the distribution (no gradient)
        # Do not use `rsample()`, it leads to zero gradient of mean head !
        action = action_dist.sample()
        # Calculate log-probability of the sampled action
        action_logprob = action_dist.log_prob(action)
        # Calculate policy entropy conditioned on state
        entropy = action_dist.entropy()
        # Calculate policy perplexity i.e. exp(entropy)
        perplexity = action_dist.perplexity()
        
        ##############################
        # TEMP: sanity check for NaN #
        ##############################
        if torch.any(torch.isnan(action_logprob)):
            while True:
                print(f'NaN, check your std: {std}')
        
        # Constraint action in valid range
        action = self.constraint_action(action)
        
        # Record output
        out_policy['action'] = action
        out_policy['action_logprob'] = action_logprob
        out_policy['entropy'] = entropy
        out_policy['perplexity'] = perplexity
        
        return out_policy
        
    def constraint_action(self, action):
        r"""Clipping the action with valid upper/lower bound defined in action space. 
        
        .. note::
        
            We assume all dimensions in continuous action space share the identical high and low value
            e.g. low = [-2.0, -2.0] and high = [2.0, 2.0]
            
        .. warning::
        
            The constraint action should be placed after computing the log-probability. It happens before
            it, the log-probability will be definitely wrong value. 
        
        Args:
            action (Tensor): action sampled from Normal distribution. 
            
        Returns
        -------
        constrained_action : Tensor
            constrained action.
        """
        # Get valid range
        low = np.unique(self.env_spec.action_space.low)
        high = np.unique(self.env_spec.action_space.high)
        assert low.ndim == 1 and high.ndim == 1, 'low and high should be identical for each dimension'
        assert -low.item() == high.item(), 'low and high should be identical with absolute value'
        
        # Clip action to value range i.e. [low, high]
        constrained_action = torch.clamp(action, min=low.item(), max=high.item())
        
        return constrained_action
