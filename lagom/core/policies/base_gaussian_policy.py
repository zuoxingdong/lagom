from .base_policy import BasePolicy

import torch

from torch.distributions import Normal
        
    
class BaseGaussianPolicy(BasePolicy):
    """
    Base class of Gaussian policy (independent) for continuous action space. 
    Action can be sampled from a Normal distribution. 
    
    Note that the user-defined network should return a dictionary
    from its forward function. At least with the key ['mean', 'logvar']. 
    It can also contain the key 'value' for the value function of actor-critic network.
    
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
    def __call__(self, x):
        network_out = self.network(x)
        assert isinstance(network_out, dict) and 'mean' in network_out and 'logvar' in network_out
        
        # Get mean and logvar for the action
        mean = network_out['mean']
        logvar = network_out['logvar']
        # Obtain std: exp(0.5*log(std**2))
        std = torch.exp(0.5*logvar)
        # Create indpendent normal distribution 
        action_dist = Normal(loc=mean, scale=std)
        # Sample an action from the distribution
        # We use PyTorch build-in reparameterized verion, rsample()
        action = action_dist.rsample()
        # Calculate log-probability of sampled action
        action_logprob = action_dist.log_prob(action)
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
