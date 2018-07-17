from .base_policy import BasePolicy

import torch.nn.functional as F

from torch.distributions import Categorical
    
    
class BaseCategoricalPolicy(BasePolicy):
    """
    Base class of categorical policy for discrete action space. 
    Action can be sampled from a categorical distribution. 
    
    Note that the user-defined network should return a dictionary
    from its forward function. At least with the key ['action_scores'] (without softmax). 
    It can also contain the key 'value' for the value function of actor-critic network.
    
    All inherited subclasses should implement the following function
    1. process_network_output(self, network_out)
    
    Examples:
    
        env = gym.make('CartPole-v0')
        env_spec = EnvSpec(GymEnv(env))
    
        class MLP(BaseMLP):
            def make_params(self, config):
                self.fc1 = nn.Linear(in_features=4, out_features=32)
                self.fc2 = nn.Linear(in_features=32, out_features=2)

            def init_params(self, config):
                gain = nn.init.calculate_gain(nonlinearity='relu')

                nn.init.orthogonal_(self.fc1.weight, gain=gain)
                nn.init.constant_(self.fc1.bias, 0.0)

                nn.init.orthogonal_(self.fc2.weight, gain=gain)
                nn.init.constant_(self.fc2.bias, 0.0)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)


                # Output dictionary
                out = {}
                out['action_scores'] = x

                return out


        class CategoricalPolicy(BaseCategoricalPolicy):
            def process_network_output(self, network_out):
                return {}
                
        network = MLP(config=None)
        policy = CategoricalPolicy(network=network, env_spec=env_spec)
    """
    def __call__(self, x):
        network_out = self.network(x)
        assert isinstance(network_out, dict) and 'action_scores' in network_out
        
        # Get action scores from the output
        action_scores = network_out['action_scores']
        # Compute actions probabilities by using softmax
        action_probs = F.softmax(action_scores, dim=-1)  # over last dimension
        # Create a categorical distribution
        action_dist = Categorical(probs=action_probs)
        # Sample an action from the distribution
        action = action_dist.sample()
        # Calculate log-probability of sampled action
        action_logprob = action_dist.log_prob(action)
        # Calculate entropy of the policy conditional on state
        entropy = action_dist.entropy()
        # Calculate perplexity of the policy, i.e. exp(entropy)
        perplexity = action_dist.perplexity()
        
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
