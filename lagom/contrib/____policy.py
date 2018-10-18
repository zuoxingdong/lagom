from .base_policy import BasePolicy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.distributions import Independent
from torch.distributions import Normal

from lagom.core.networks import ortho_init
    
    
class CategoricalPolicy(BasePolicy):
    def __init__(self, config, network, env_spec, device, learn_V=False, **kwargs):
        super().__init__(config=config, network=network, env_spec=env_spec, device=device, **kwargs)
        if self.recurrent:
            self.reset_rnn_states()
        
    def __call__(self, x, out_keys=['action'], info={}, **kwargs):
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
        
        if 'state_value' in out_keys:
            out_policy['state_value'] = self.network.value_head(features).squeeze(-1)  # squeeze final single dim
        
        action_prob = F.softmax(action_score, dim=-1)  # over last dimension
        if 'action_prob' in out_keys:
            out_policy['action_prob'] = action_prob
        
        action = action_dist.sample()
        out_policy['action'] = action
        
        if 'action_logprob' in out_keys:
            out_policy['action_logprob'] = action_dist.log_prob(action)
            
        if 'entropy' in out_keys:
            out_policy['entropy'] = action_dist.entropy()
        
        if 'perplexity' in out_keys:
            out_policy['perplexity'] = action_dist.perplexity()
        
        return out_policy

class GaussianPolicy(BasePolicy):
    def __init__(self,
                 config,
                 network, 
                 env_spec, 
                 device,
                 learn_V=False,
                 min_std=1e-6, 
                 std_style='exp', 
                 constant_std=None,
                 std_state_dependent=False,
                 init_std=1.0,
                 **kwargs):
        super().__init__(config=config, network=network, env_spec=env_spec, device=device, **kwargs)
        
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
            
        # Forward pass of value head to obtain value function if required
        if 'state_value' in out_keys:
            out_policy['state_value'] = self.network.value_head(features).squeeze(-1)  # squeeze final single dim
        # Sample action from the distribution (no gradient)
        # Do not use `rsample()`, it leads to zero gradient of mean head !
        action = action_dist.sample()
        out_policy['action'] = action
        
        if 'action_logprob' in out_keys:
            out_policy['action_logprob'] = action_dist.log_prob(action)
        
        if 'entropy' in out_keys:
            out_policy['entropy'] = action_dist.entropy()
        
        if 'perplexity' in out_keys:
            out_policy['perplexity'] = action_dist.perplexity()
        
        # sanity check for NaN
        if torch.any(torch.isnan(action)):
            while True:
                msg = 'NaN ! A workaround is to learn state-independent std or use tanh rather than relu'
                msg2 = f'check: \n\t mean: {mean}, logvar: {logvar}'
                print(msg + msg2)
        
        out_policy['action'] = self.constraint_action(action)
        
        return out_policy
        
    
