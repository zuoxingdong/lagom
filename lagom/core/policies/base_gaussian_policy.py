from .base_policy import BasePolicy

import numpy as np

import torch
import torch.nn.functional as F

from torch.distributions import Normal
        
    
class BaseGaussianPolicy(BasePolicy):
    """
    Base class of Gaussian policy (independent) for continuous action space. 
    Action can be sampled from a Normal distribution. 
    
    Note that the user-defined network should return a dictionary
    from its forward function. At least with the key ['mean', 'logvar']. 
    It can also contain the key 'value' for the value function of actor-critic network.
    
    Depending on the cases, the standard deviation can be state-dependent or state-independent.
        - state-dependent: the std trainable parameters are connected from the last output layer.
            e.g. logvar_head = nn.Linear(in_features=64, out_features=4)
        - state-independent: the std trainable parameters are exclusive from policy network parameters. 
            e.g. logvar_head = nn.Parameter(torch.full([4], -4.6))  # -4.6 = log(0.1**2) for 0.1 as init std
    Sometimes, it can also be useful to have constant standard deviation, it is supported with this class. 
    
    All inherited subclasses should at least implement the following function
    1. process_network_output(self, network_out)
    2. constraint_action(self, action)
    
    Examples:
    
        env = gym.make('Pendulum-v0')
        env_spec = EnvSpec(GymEnv(env))
    
        class MLP(BaseMLP):
            def make_params(self, config):
                self.fc1 = nn.Linear(in_features=3, out_features=32)

                self.mean_head = nn.Linear(in_features=32, out_features=1)
                self.logvar_head = nn.Linear(in_features=32, out_features=1)

            def init_params(self, config):
                gain = nn.init.calculate_gain(nonlinearity='relu')

                nn.init.orthogonal_(self.fc1.weight, gain=gain)
                nn.init.constant_(self.fc1.bias, 0.0)

                nn.init.orthogonal_(self.mean_head.weight, gain=gain)
                nn.init.constant_(self.mean_head.bias, 0.0)

                nn.init.orthogonal_(self.logvar_head.weight, gain=gain)
                nn.init.constant_(self.logvar_head.bias, 0.0)

            def forward(self, x):
                x = F.relu(self.fc1(x))

                mean = self.mean_head(x)
                logvar = self.logvar_head(x)

                # Output dictionary
                out = {}
                out['mean'] = mean
                out['logvar'] = logvar

                return out


        class GaussianPolicy(BaseGaussianPolicy):
            def process_network_output(self, network_out):
                return {}

            def constraint_action(self, action):
                return 2*torch.tanh(action)
                
        network = MLP(config=None)
        policy = GaussianPolicy(network=network, env_spec=env_spec)
    """
    def __init__(self,
                 network, 
                 env_spec,
                 config,
                 min_std=1e-6, 
                 std_style='exp', 
                 constant_std=None, 
                 **kwargs):
        """
        Args:
            network (BaseNetwork): an instantiated user-defined network. 
            env_spec (EnvSpec): environment specification. 
            config (dict): A dictionary for the configuration. 
            min_std (float): minimum threshould for standard deviation to avoid numerical instability. 
            std_style (str): parameterization for standard deviation.  
                - 'exp': assume network outputs log-variance, and apply exp(0.5*logvar)
                - 'softplus': assume network outputs raw variance logits, and apply sqrt(log(1 + exp(x)))
            constant_std (ndarray): An array of constant standard deviation for all dimensions. 
                If it is not None, then it will use constant std and will ignore whether or not network has
                trainable logvar. 
            **kwargs: keyword aguments used to specify the policy
        """
        super().__init__(network=network, env_spec=env_spec, config=config, **kwargs)
        
        self.min_std = min_std
        self.std_style = std_style
        self.constant_std = constant_std
    
    def __call__(self, x):
        network_out = self.network(x)
        assert isinstance(network_out, dict) and 'mean' in network_out
        
        # Get mean and std for the action distribution
        mean = network_out['mean']
        if 'logvar' in network_out:  # network has trainable logvar
            logvar = network_out['logvar']
            if self.std_style == 'exp':
                std = torch.exp(0.5*logvar)
            elif self.std_style == 'softplus':
                std = torch.sqrt(F.softplus(logvar, beta=5))
        else:  # network does not have trainable logvar, use constant std instead
            assert self.constant_std is not None
            std = torch.from_numpy(np.array(self.constant_std)).type_as(mean)
            std = std.to(mean.device)
        
        # Constraint lower bound of std to avoid numerical instability
        min_std = torch.full(std.size(), self.min_std).type_as(std).to(std.device)
        std = torch.max(std, min_std)
            
        # Create indpendent normal distribution 
        action_dist = Normal(loc=mean, scale=std)
        # Sample an action from the distribution
        # We use PyTorch build-in reparameterized verion, rsample()
        action = action_dist.rsample()
        # Calculate log-probability of sampled action
        action_logprob = action_dist.log_prob(action)
        
        
        
        
        
        
        
        
        if torch.any(torch.isnan(action_logprob)):
            print(f'action: {action}, std: {std}, logvar: {logvar}')
            [print('@'*5000) for _ in range(500000)]
            raise ValueError
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            
        # Calculate entropy of the policy conditional on state
        entropy = action_dist.entropy()
        # Calculate perplexity of the policy, i.e. exp(entropy)
        perplexity = action_dist.perplexity()
        
        # Constraint action with lower/upper bounds
        # TODO: where should we put before/after logprob ?
        # https://discuss.pytorch.org/t/should-action-log-probability-computed-after-or-before-constraining-the-action/20976
        # Note that it will be completely wrong if put constraint transformation
        # before computing the log-probability. Because log-prob with transformed action is 
        # definitely a wrong value, it's equivalent to transformation of a Gaussian distribution
        # and compute transformed samples with Gaussian density. 
        action = self.constraint_action(action)
        
        # User-defined function to process any possible other output
        processed_network_out = self.process_network_output(network_out)
        
        # Dictionary of output
        out = {}
        out['action'] = action
        out['action_logprob'] = action_logprob
        out['entropy'] = entropy
        out['perplexity'] = perplexity
        # Augment with dictionary returned from processed network output
        out = {**out, **processed_network_out}
        
        return out
    
    def constraint_action(self, action):
        """
        User-defined function to smoothly constraint the action with upper/lower bounds. 
        
        The constraint must be smooth (differentiable), it is recommended to use functions
        like tanh, or sigmoid. For example the action is in the range of [-2, 2], one can define
        `constrained_action = 2*torch.tanh(action)`. 
        
        If there is no need to constraint, then it is required to send the action back. 
        i.e. `return action`
        
        Args:
            action (Tensor): action sampled from Normal distribution. 
            
        Returns:
            constrained_action (Tensor): constrained action. 
        """
        raise NotImplementedError
