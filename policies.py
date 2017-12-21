import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
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
