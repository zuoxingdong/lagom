from lagom.networks import BaseNetwork

import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from lagom.networks import ortho_init


class CategoricalHead(BaseNetwork):
    r"""Defines a categorical neural network head for discrete action space.  
    
    Example::
    
        >>> from lagom.envs import make_gym_env, EnvSpec
        >>> env = make_gym_env('CartPole-v1', 0)
        >>> env_spec = EnvSpec(env)

        >>> head = CategoricalHead(None, None, 30, env_spec)
        >>> head(torch.randn(3, 30))
        Categorical(probs: torch.Size([3, 2]))
    
    """
    def __init__(self, config, device, feature_dim, env_spec, **kwargs):
        self.feature_dim = feature_dim
        self.env_spec = env_spec
        assert self.env_spec.control_type == 'Discrete', 'expected as Discrete control type'
        
        super().__init__(config, device, **kwargs)
        
        
    def make_params(self, config):
        self.action_head = nn.Linear(in_features=self.feature_dim, 
                                     out_features=self.env_spec.action_space.flat_dim)
        
        
    def init_params(self, config):
        # 0.01->uniformly distributed
        ortho_init(self.action_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)
        
    def reset(self, config, **kwargs):
        pass
    
    def forward(self, x):
        action_score = self.action_head(x)
        action_prob = F.softmax(action_score, dim=-1)
        action_dist = Categorical(probs=action_prob)
        
        return action_dist
