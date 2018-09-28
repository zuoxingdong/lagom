from .base_policy import BasePolicy

import numpy as np

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
        
    """
    def __init__(self, config, network, env_spec, device, learn_V=False, **kwargs):
        super().__init__(config=config, network=network, env_spec=env_spec, device=device, **kwargs)
        self.learn_V = learn_V
        
        assert self.env_spec.control_type == 'Discrete', 'expected as Discrete control type'
        assert hasattr(self.network, 'last_feature_dim'), 'network expected to have an attribute `.last_feature_dim`'
        
        # Create action head, orthogonal initialization and put onto device
        action_head = nn.Linear(in_features=self.network.last_feature_dim, 
                                out_features=self.action_space.flat_dim)
        ortho_init(action_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)  # 0.01->uniformly distributed
        action_head = action_head.to(self.device)
        # Augment to network (e.g. tracked by network.parameters() for optimizer to update)
        self.network.add_module('action_head', action_head)
        
        # Create value head (if required), orthogonal initialization and put onto device
        if self.learn_V:
            value_head = nn.Linear(in_features=self.network.last_feature_dim, out_features=1)
            ortho_init(value_head, nonlinearity=None, weight_scale=1.0, constant_bias=0.0)
            value_head = value_head.to(self.device)
            self.network.add_module('value_head', value_head)
            
        # Initialize and track the RNN hidden states
        if self.recurrent:
            self.reset_rnn_states()
        
    def __call__(self, x, out_keys=['action'], info={}, **kwargs):
        # Output dictionary
        out_policy = {}
        
        # Forward pass of feature networks to obtain features
        if self.recurrent:
            if 'mask' in info:  # make the mask
                mask = np.logical_not(info['mask']).astype(np.float32)
                mask = torch.from_numpy(mask).unsqueeze(1).to(self.device)
            else:
                mask = None
                
            out_network = self.network(x=x, 
                                       hidden_states=self.rnn_states, 
                                       mask=mask)
            features = out_network['output']
            # Update the tracking of current RNN hidden states
            if 'rnn_state_no_update' not in info:
                self.rnn_states = out_network['hidden_states']
        else:
            features = self.network(x)
        
        # Forward pass of action head to obtain action scores for categorical distribution
        action_score = self.network.action_head(features)
        
        # Forward pass of value head to obtain value function if required
        if 'state_value' in out_keys:
            out_policy['state_value'] = self.network.value_head(features).squeeze(-1)  # squeeze final single dim
        
        # Compute action probabilities by applying softmax
        action_prob = F.softmax(action_score, dim=-1)  # over last dimension
        if 'action_prob' in out_keys:
            out_policy['action_prob'] = action_prob
            
        # Create a Categorical distribution
        action_dist = Categorical(probs=action_prob)
        
        # Sample action from the distribution (no gradient)
        action = action_dist.sample()
        out_policy['action'] = action
        
        # Calculate log-probability of the sampled action
        if 'action_logprob' in out_keys:
            out_policy['action_logprob'] = action_dist.log_prob(action)
            
        # Calculate policy entropy conditioned on state
        if 'entropy' in out_keys:
            out_policy['entropy'] = action_dist.entropy()
        
        # Calculate policy perplexity i.e. exp(entropy)
        if 'perplexity' in out_keys:
            out_policy['perplexity'] = action_dist.perplexity()
        
        return out_policy
