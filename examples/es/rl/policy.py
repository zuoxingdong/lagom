import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.agents import BaseAgent

from lagom.core.networks import BaseNetwork
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
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        return x
    

class Agent(BaseAgent):
    def choose_action(self, obs):
        obs = torch.from_numpy(np.asarray(obs)).float()  # already batched due to VecEnv
        
        with torch.no_grad():
            out_policy = self.policy(obs, out_keys=['action'])
        
        return out_policy
