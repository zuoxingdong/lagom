import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.core.networks import BaseNetwork
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
