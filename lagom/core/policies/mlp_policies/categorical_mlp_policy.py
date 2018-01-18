import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from lagom.core.policies.mlp_policies import BaseMLPPolicy


class CategoricalMLPPolicy(BaseMLPPolicy):
    def __init__(self, env_spec, fc_sizes=[128], predict_value=False):
        self.env_spec = env_spec
        self.fc_sizes = fc_sizes
        self.predict_value = predict_value
        
        super().__init__(self.env_spec, self.fc_sizes, self.predict_value)
        
    def _process_input(self, x):
        # Unpack input data
        obs = x.get('observation', None)
        # Convert observation into single input vector
        x = np.array(obs).reshape(-1)
        
        # Convert data into FloatTensor and Variable with batch dimension
        x = Variable(torch.FloatTensor(x).unsqueeze(0))
        
        return x
        
        
class CategoricalMLPGoalPolicy(BaseMLPPolicy):
    def __init__(self, env_spec, fc_sizes=[128], predict_value=False):
        self.env_spec = env_spec
        self.fc_sizes = fc_sizes
        self.predict_value = predict_value
        
        super().__init__(self.env_spec, self.fc_sizes, self.predict_value)
        
        # Goal-conditional policy: augment first layer with goal as additional input
        in_features = self.fc_layers.fc0.in_features + self.env_spec.get('goal_dim')
        out_features = self.fc_layers.fc0.out_features
        self.fc_layers.fc0 = nn.Linear(in_features, out_features)
        
    def _process_input(self, x):
        # Unpack input data
        obs = x.get('observation', None)
        goal = x.get('goal_state', None)
        
        # Convert observation and goal into single input vector
        x = np.array(obs).reshape(-1)
        goal = np.array(goal).reshape(-1)
        # Concatenate input data
        x = np.concatenate([x, goal])
        
        # Convert data into FloatTensor and Variable with batch dimension
        x = Variable(torch.FloatTensor(x).unsqueeze(0))
        
        return x