from lagom.networks import BaseNetwork

import torch.nn as nn

from lagom.networks import ortho_init


class StateValueHead(BaseNetwork):
    r"""Defines a neural network head for state value function.
    
    Example:
    
        >>> value_head = StateValueHead(None, None, 30)
        >>> value_head(torch.randn(3, 30))
        tensor([[ 0.2689],
        [ 0.7674],
        [-0.7288]], grad_fn=<ThAddmmBackward>)
        
    """
    def __init__(self, config, device, feature_dim, **kwargs):
        self.feature_dim = feature_dim
        
        super().__init__(config, device, **kwargs)
        
    def make_params(self, config):
        self.value_head = nn.Linear(in_features=self.feature_dim, out_features=1)
        
    def init_params(self, config):
        ortho_init(self.value_head, nonlinearity=None, weight_scale=1.0, constant_bias=0.0)
        
    def reset(self, config, **kwargs):
        pass
    
    def forward(self, x):
        state_value = self.value_head(x)
        
        return state_value
