import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Categorical Policy
class MLPPolicy(nn.Module):
    def __init__(self, env_spec, hidden_neuron):
        super().__init__()
        
        self.policy_type='classic'
        
        self.fc1 = nn.Linear(env_spec['input_dim'], hidden_neuron)
        self.fc2 = nn.Linear(hidden_neuron, env_spec['action_dim'])
        
    def forward(self, x):
        # Classic policy, unpack observation data
        x = x['observation']
        # Flatten observation to single vector
        x = np.array(x).reshape(-1)
        # Convert input data into FloatTensor and Variable with batch dimension
        x = Variable(torch.FloatTensor(x)).unsqueeze(0)
        
        # Forward pass
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
        
        return F.softmax(action_scores, dim=1)
    
class MLPGoalPolicy(nn.Module):
    def __init__(self, env_spec, hidden_neuron):
        super().__init__()
        
        self.policy_type = 'goal'
        
        self.fc1 = nn.Linear(env_spec['input_dim'] + env_spec['goal_dim'], hidden_neuron)
        self.fc2 = nn.Linear(hidden_neuron, env_spec['action_dim'])
    
    def forward(self, x):
        # Goal-conditional policy, unpack data
        obs = x['observation']
        goals = x['goal_state']
        # Flatten data to single vector
        obs = np.array(obs).reshape(-1)
        goals = np.array(goals).reshape(-1)
        # Concatenate different input data into single vector
        x = np.concatenate([obs, goals])
        # Convert input data into FloatTensor and Variable with batch dimension
        x = Variable(torch.FloatTensor(x)).unsqueeze(0)
        
        # Forward pass
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
        
        return F.softmax(action_scores, dim=1)
        

class MLPValuePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        
        return F.softmax(action_scores, dim=1), state_values
