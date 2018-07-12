import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.core.networks import BaseMLP
from lagom.core.policies import BaseCategoricalPolicy


class MLP(BaseMLP):
    def make_params(self, config):
        self.fc1 = nn.Linear(in_features=4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)
        
    def init_params(self, config):
        gain = nn.init.calculate_gain(nonlinearity='relu')

        nn.init.orthogonal_(self.fc1.weight, gain=gain)
        nn.init.constant_(self.fc1.bias, 0.0)

        nn.init.orthogonal_(self.fc2.weight, gain=gain)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
        
        # Output dictionary
        out = {}
        out['action_scores'] = action_scores

        return out
    

class CategoricalPolicy(BaseCategoricalPolicy):
    def process_network_output(self, network_out):
        return {}