import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.agents import BaseAgent

from lagom.core.networks import BaseNetwork
from lagom.core.networks import BaseRNN
from lagom.core.networks import make_fc
from lagom.core.networks import ortho_init

from lagom.core.policies import CategoricalPolicy
from lagom.core.policies import GaussianPolicy


class Network(BaseNetwork):
    def make_params(self, config):
        self.layers = make_fc(input_dim=self.env_spec.observation_space.flat_dim, 
                              hidden_sizes=config['network.hidden_size'])
        self.last_feature_dim = config['network.hidden_size'][-1]

    def init_params(self, config):
        for layer in self.layers:
            ortho_init(layer, nonlinearity='tanh', constant_bias=0.0)

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))

        return x
    
    
class LSTM(BaseRNN):
    def make_params(self, config):
        self.rnn = nn.LSTMCell(input_size=self.env_spec.observation_space.flat_dim, 
                               hidden_size=config['network.hidden_size'][0])

        self.last_feature_dim = config['network.hidden_size'][0]
        
    def init_params(self, config):
        ortho_init(self.rnn, nonlinearity=None, weight_scale=1.0, constant_bias=0.0)
        
    def init_hidden_states(self, config, batch_size, **kwargs):
        h = torch.zeros(batch_size, config['network.hidden_size'][0])
        h = h.to(self.device)
        c = torch.zeros_like(h)

        return [h, c]
    
    def rnn_forward(self, x, hidden_states, mask=None, **kwargs):
        # mask out hidden states if required
        if mask is not None:
            h, c = hidden_states
            mask = mask.to(self.device)
            
            h = h*mask
            c = c*mask
            hidden_states = [h, c]

        h, c = self.rnn(x, hidden_states)

        out = {'output': h, 'hidden_states': [h, c]}

        return out
    

class Agent(BaseAgent):
    def __init__(self, config, policy, **kwargs):
        super().__init__(config, **kwargs)
        
        self.policy = policy
    
    def choose_action(self, obs, info={}):
        # Reset RNN states if required
        if self.policy.recurrent and self.info['reset_rnn_states']:
            self.policy.reset_rnn_states()
            self.info['reset_rnn_states'] = False  # Already reset, so turn off
        
        # Convert batched observation
        obs = torch.from_numpy(np.asarray(obs)).float().to(self.policy.device)
        
        with torch.no_grad():
            out_policy = self.policy(obs, out_keys=['action'], info=info)
        
        return out_policy
    
    def learn(self, D, info={}):
        pass
    
    def save(self, f):
        pass
    
    def load(self, f):
        pass
