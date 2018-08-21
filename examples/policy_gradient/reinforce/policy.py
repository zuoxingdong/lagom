import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.core.networks import BaseMLP
from lagom.core.policies import BaseCategoricalPolicy
from lagom.core.policies import BaseGaussianPolicy


class CategoricalMLP(BaseMLP):
    def make_params(self, config):
        self.fc1 = nn.Linear(in_features=self.env_spec.observation_space.flat_dim, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        
        self.action_head = nn.Linear(in_features=64, out_features=self.env_spec.action_space.flat_dim)
        
    def init_params(self, config):
        gain = nn.init.calculate_gain(nonlinearity='tanh')

        nn.init.orthogonal_(self.fc1.weight, gain=gain)
        nn.init.constant_(self.fc1.bias, 0.0)
        
        nn.init.orthogonal_(self.fc2.weight, gain=gain)
        nn.init.constant_(self.fc2.bias, 0.0)

        nn.init.orthogonal_(self.action_head.weight, gain=0.01)  # Smaller scale for action head
        nn.init.constant_(self.action_head.bias, 0.0)

    def forward(self, x):
        # Output dictionary
        network_out = {}
        
        # Flatten the input
        x = x.flatten(start_dim=1)
        
        # Forward pass through feature layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # Forward pass through action layers and record the output
        action_scores = self.action_head(x)
        network_out['action_scores'] = action_scores
        
        return network_out


class GaussianMLP(BaseMLP):
    def make_params(self, config):
        self.fc1 = nn.Linear(in_features=self.env_spec.observation_space.flat_dim, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        
        self.mean_head = nn.Linear(in_features=64, out_features=self.env_spec.action_space.flat_dim)
        if config['agent:constant_std'] is None:  # no constant std provided, so train it
            if config['agent:std_state_dependent']:  # std is dependent on state
                self.logvar_head = nn.Linear(in_features=64, out_features=self.env_spec.action_space.flat_dim)
            else:  # std is independent of state
                # Do not initialize it in `init_params()`
                self.logvar_head = nn.Parameter(torch.full([self.env_spec.action_space.flat_dim], 0.01))

    def init_params(self, config):
        gain = nn.init.calculate_gain(nonlinearity='tanh')

        nn.init.orthogonal_(self.fc1.weight, gain=gain)
        nn.init.constant_(self.fc1.bias, 0.0)
        
        nn.init.orthogonal_(self.fc2.weight, gain=gain)
        nn.init.constant_(self.fc2.bias, 0.0)

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)  # small initial mean around 0. 
        nn.init.constant_(self.mean_head.bias, 0.0)
        if config['agent:constant_std'] is None and config['agent:std_state_dependent']:
            nn.init.orthogonal_(self.logvar_head.weight, gain=0.01)
            nn.init.constant_(self.logvar_head.bias, 0.0)

    def forward(self, x):
        # Output dictionary
        network_out = {}
        
        # Flatten the input
        x = x.flatten(start_dim=1)
        
        # Forward pass through feature layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # Forward pass through action layers and record the output
        mean = self.mean_head(x)
        network_out['mean'] = mean
        
        if self.config['agent:constant_std'] is None:  # learned std
            if self.config['agent:std_state_dependent']:  # state-dependent std, so forward pass
                logvar = self.logvar_head(x)
            else:  # state-independent, so directly use it
                logvar = self.logvar_head.expand_as(mean)
            network_out['logvar'] = logvar

        return network_out
    

class CategoricalPolicy(BaseCategoricalPolicy):
    def process_network_output(self, network_out):
        return network_out


class GaussianPolicy(BaseGaussianPolicy):
    def process_network_output(self, network_out):
        return network_out

    def constraint_action(self, action):
        # Limit the action with valid range
        # Note that we assume all Continuous action space with same low and same high for each dimension
        # and asymmetric (absolute values between low and high should be identical)
        low = np.unique(self.env_spec.action_space.low)
        high = np.unique(self.env_spec.action_space.high)
        assert low.ndim == 1 and high.ndim == 1
        assert -low.item() == high.item()
        
        # Enforce valid action in [low, high]
        action = torch.clamp(action, min=low.item(), max=high.item())
        
        return action
