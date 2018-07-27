import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.core.networks import BaseMLP
from lagom.core.policies import BaseCategoricalPolicy


class MLP(BaseMLP):
    def make_params(self, config):
        self.fc1 = nn.Linear(in_features=4, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        
        self.action_head = nn.Linear(in_features=64, out_features=2)
        self.value_head = nn.Linear(in_features=64, out_features=1)
        
    def init_params(self, config):
        gain = nn.init.calculate_gain(nonlinearity='relu')

        nn.init.orthogonal_(self.fc1.weight, gain=gain)
        nn.init.constant_(self.fc1.bias, 0.0)
        
        nn.init.orthogonal_(self.fc2.weight, gain=gain)
        nn.init.constant_(self.fc2.bias, 0.0)

        nn.init.orthogonal_(self.action_head.weight, gain=0.01)  # Smaller scale for action head
        nn.init.constant_(self.action_head.bias, 0.0)
        
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)  # no nonlinearity
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x):
        # Output dictionary
        out = {}
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_scores = self.action_head(x)
        out['action_scores'] = action_scores
        
        state_value = self.value_head(x)
        out['state_value'] = state_value

        return out
    

class CategoricalPolicy(BaseCategoricalPolicy):
    def process_network_output(self, network_out):
        return network_out
