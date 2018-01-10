import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


#################################
# TODO: Continuous MLP policy (Categorical/Gaussian)

# utils.py function of make_policy(args.....)
#################################


# Base class for MLP policies
class BaseMLPPolicy(nn.Module):
    def __init__(self, env_spec, fc_sizes=[128]):
        """
        Base class for fully connected network (or Multi-Layer Perceptron)
        
        Args:
            env_spec: A list of specifications of the environment.
            fc_sizes: A list of number of hidden neurons for fully connected layer.
        """
        super().__init__()
        
        # Iteratively build network, should use nn.Sequential, otherwise cannot be recognized
        self.fc_layers = nn.Sequential()
        for i, size in enumerate(fc_sizes):
            if i == 0:  # first hidden layer
                in_features = env_spec['obs_dim']
            else:  # number of out_features from previous layer as in_features for current layer
                in_features = self.fc_layers[i-1].out_features
            # Add FC layer
            self.fc_layers.add_module('fc' + str(i), nn.Linear(in_features, size))
            
        # Action layer
        in_features = fc_sizes[-1]
        out_features = env_spec['action_dim']
        self.action_head = nn.Linear(in_features, out_features)
        

class CategoricalMLPPolicy(BaseMLPPolicy):
    def __init__(self, env_spec, fc_sizes=[128]):
        self.env_spec = env_spec
        self.fc_sizes = fc_sizes
        
        super().__init__(self.env_spec, self.fc_sizes)
        
    def forward(self, x):
        # Unpack input data
        obs = x.get('observation', None)
        # Convert observation into single input vector
        x = np.array(obs).reshape(-1)
        
        # Convert data into FloatTensor and Variable with batch dimension
        x = Variable(torch.FloatTensor(x).unsqueeze(0))
        
        # Forward pass till final hidden layer
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        # Action head layer
        action_scores = self.action_head(x)
        action_probs = F.softmax(action_scores, dim=1)
        
        # Output dictionary
        output = {}
        output['action_probs'] = action_probs
        
        return output


class CategoricalMLPValuePolicy(BaseMLPPolicy):
    def __init__(self, env_spec, fc_sizes=[128]):
        self.env_spec = env_spec
        self.fc_sizes = fc_sizes
        
        super().__init__(self.env_spec, self.fc_sizes)
        
        # Value layer
        in_features = fc_sizes[-1]
        out_features = 1
        self.value_head = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # Unpack input data
        obs = x.get('observation', None)
        # Convert observation into single input vector
        x = np.array(obs).reshape(-1)
        
        # Convert data into FloatTensor and Variable with batch dimension
        x = Variable(torch.FloatTensor(x).unsqueeze(0))
        
        # Forward pass till final hidden layer
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        # Action head layer
        action_scores = self.action_head(x)
        action_probs = F.softmax(action_scores, dim=1)
        # Value head layer
        state_value = self.value_head(x)
        
        # Output dictionary
        output = {}
        output['action_probs'] = action_probs
        output['state_value'] = state_value
        
        return output
        
        
class CategoricalMLPGoalPolicy(BaseMLPPolicy):
    def __init__(self, env_spec, fc_sizes=[128]):
        self.env_spec = env_spec
        self.fc_sizes = fc_sizes
        
        super().__init__(self.env_spec, self.fc_sizes)
        
        # Goal-conditional policy
        in_features = self.fc_layers.fc0.in_features + self.env_spec['goal_dim']
        out_features = self.fc_layers.fc0.out_features
        self.fc_layers.fc0 = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # Unpack input data
        obs = x.get('observation', None)
        goals = x.get('goal_state', None)
        
        # Convert observation and goals into single input vector
        x = np.array(obs).reshape(-1)
        goals = np.array(goals).reshape(-1)
        # Concatenate input data
        x = np.concatenate([x, goals])
        
        # Convert data into FloatTensor and Variable with batch dimension
        x = Variable(torch.FloatTensor(x).unsqueeze(0))
        
        # Forward pass till final hidden layer
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        # Action head layer
        action_scores = self.action_head(x)
        action_probs = F.softmax(action_scores, dim=1)
        
        # Output dictionary
        output = {}
        output['action_probs'] = action_probs
        
        return output