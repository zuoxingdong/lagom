from .base_policy import BasePolicy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from lagom.core.networks import ortho_init
    
    
class CategoricalPolicy(BasePolicy):
    r"""A parameterized policy defined as a categorical distribution over a discrete action space. 
    
    .. note::
    
        The neural network given to the policy should define all but the final output layer. The final
        output layer for the categorical distribution will be created with the policy and augmented
        to the network. This decoupled design makes it more flexible to use for different discrete 
        action spaces. Note that the network must have an attribute ``.last_feature_dim`` of type
        ``int`` for the policy to create the final output layer (fully-connected), and this is
        recommended to be done in the method :meth:`~BaseNetwork.make_params` of the network class.
    
    Example::

        >>> policy = CategoricalPolicy(config=config, network=network, env_spec=env_spec)
        >>> policy(observation)
        
    """
    def __init__(self, config, network, env_spec, **kwargs):
        super().__init__(config=config, network=network, env_spec=env_spec, **kwargs)
        
        assert self.env_spec.control_type == 'Discrete', 'expected as Discrete control type'
        assert hasattr(self.network, 'last_feature_dim'), 'network expected to have an attribute `.last_feature_dim`'
        
        # Create final output layer
        action_head = nn.Linear(in_features=self.network.last_feature_dim, 
                                out_features=self.env_spec.action_space.flat_dim)
        # Orthogonal initialization to the parameters with scale 0.01, i.e. uniformly distributed
        ortho_init(action_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)
        # Augment to network (e.g. tracked by network.parameters() for optimizer to update)
        self.network.add_module('action_head', action_head)
        
    def __call__(self, x):
        # Output dictionary
        out_policy = {}
        
        # Forward pass through neural network for the input
        features = self.network(x)
        
        # Forward pass of action head to obtain action scores for categorical distribution
        action_score = self.network.action_head(features)
        # Compute action probabilities by applying softmax
        action_prob = F.softmax(action_score, dim=-1)  # over last dimension
        # Create a Categorical distribution
        action_dist = Categorical(probs=action_prob)
        # Sample action from the distribution (no gradient)
        action = action_dist.sample()
        # Calculate log-probability of the sampled action
        action_logprob = action_dist.log_prob(action)
        # Calculate policy entropy conditioned on state
        entropy = action_dist.entropy()
        # Calculate policy perplexity i.e. exp(entropy)
        perplexity = action_dist.perplexity()
        
        # Record output
        out_policy['action'] = action
        out_policy['action_prob'] = action_prob
        out_policy['action_logprob'] = action_logprob
        out_policy['entropy'] = entropy
        out_policy['perplexity'] = perplexity
        
        return out_policy
