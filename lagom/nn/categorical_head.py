import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .module import Module
from .init import ortho_init


class CategoricalHead(Module):
    r"""Defines a module for a Categorical (discrete) action distribution. 
    
    Example:
    
        >>> import torch
        >>> action_head = CategoricalHead(30, 4, 'cpu')
        >>> action_head(torch.randn(2, 30))
        Categorical(probs: torch.Size([2, 4]))
        
    Args:
        feature_dim (int): number of input features
        num_action (int): number of discrete actions
        **kwargs: keyword arguments for more specifications.
    
    """
    def __init__(self, feature_dim, num_action, **kwargs):
        super().__init__(**kwargs)
        
        self.feature_dim = feature_dim
        self.num_action = num_action
        
        self.action_head = nn.Linear(self.feature_dim, self.num_action)
        # weight_scale=0.01 -> uniformly distributed
        ortho_init(self.action_head, weight_scale=0.01, constant_bias=0.0)

    def forward(self, x):
        action_score = self.action_head(x)
        action_prob = F.softmax(action_score, dim=-1)
        action_dist = Categorical(probs=action_prob)
        return action_dist
