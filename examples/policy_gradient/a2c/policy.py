import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.core.networks import BaseNetwork
from lagom.core.networks import BaseRNN
from lagom.core.networks import LayerNormLSTMCell
from lagom.core.networks import make_fc
from lagom.core.networks import ortho_init


class Network(BaseNetwork):
    def make_params(self, config):
        self.layers = make_fc(input_dim=self.env_spec.observation_space.flat_dim, 
                              hidden_sizes=config['network.hidden_sizes'])
        self.last_feature_dim = config['network.hidden_sizes'][-1]
        
    def init_params(self, config):
        for layer in self.layers:
            ortho_init(layer, nonlinearity='tanh', constant_bias=0.0)
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))

        return x


class LSTM(BaseRNN):
    def make_params(self, config):
        # nn.LSTMCell
        self.rnn = LayerNormLSTMCell(input_size=self.env_spec.observation_space.flat_dim, 
                               hidden_size=config['network.hidden_sizes'][0])  # TODO: support multi-layer
        self.last_feature_dim = config['network.hidden_sizes'][-1]
        
    def init_params(self, config):
        ortho_init(self.rnn, nonlinearity=None, weight_scale=1.0, constant_bias=0.0)
    
    def init_hidden_states(self, config, batch_size, **kwargs):
        h = torch.zeros(batch_size, config['network.hidden_sizes'][0])
        h = h.to(self.device)
        c = torch.zeros_like(h)

        return [h, c]
        
    def rnn_forward(self, x, hidden_states, mask=None, **kwargs):
        if mask is not None:
            mask = mask.to(self.device)
            
            h, c = hidden_states
            h = h*mask
            c = c*mask
            hidden_states = [h, c]
            
        h, c = self.rnn(x, hidden_states)
        
        out = {'output': h, 'hidden_states': [h, c]}

        return out
