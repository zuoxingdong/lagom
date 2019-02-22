import torch.nn as nn

from .module import Module
from .init import ortho_init


class StateValueHead(Module):
    r"""Defines a module for the state value function. 
    
    Example:
    
        >>> import torch
        >>> value_head = StateValueHead(10, torch.device('cpu'))
        >>> value_head(torch.randn(3, 10))
        tensor([[1.9796],
                [1.0219],
                [0.6150]], grad_fn=<ThAddmmBackward>)
    
    """
    def __init__(self, feature_dim, device, **kwargs):
        super().__init__(**kwargs)
        
        self.feature_dim = feature_dim
        self.device = device
        
        self.value_head = nn.Linear(self.feature_dim, 1)
        ortho_init(self.value_head, weight_scale=1.0, constant_bias=0.0)
        
        self.to(self.device)
        
    def forward(self, x):
        state_value = self.value_head(x)
        return state_value
