import torch
import torch.nn as nn
import torch.nn.functional as F

import gym.spaces as spaces
import lagom
import lagom.utils as utils


class Agent(lagom.BaseAgent):
    def __init__(self, config, env, **kwargs):
        super().__init__(config, env, **kwargs)
        
        self.feature_layers = lagom.nn.make_fc(spaces.flatdim(env.observation_space), config['nn.sizes'])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in config['nn.sizes']])
        feature_dim = config['nn.sizes'][-1]
        if isinstance(env.action_space, spaces.Discrete):
            self.action_head = lagom.nn.CategoricalHead(feature_dim, env.action_space.n, **kwargs)
        elif isinstance(env.action_space, spaces.Box):
            self.action_head = lagom.nn.DiagGaussianHead(feature_dim, spaces.flatdim(env.action_space), 0.5, **kwargs)

        self.register_buffer('total_timestep', torch.tensor(0))
        
    def choose_action(self, x, **kwargs):
        x = torch.as_tensor(x.observation).float().to(self.config.device).unsqueeze(0)
        for layer, layer_norm in zip(self.feature_layers, self.layer_norms):
            x = F.gelu(layer_norm(layer(x)))
        action_dist = self.action_head(x)
        action = action_dist.sample()
        out = {}
        out['raw_action'] = utils.numpify(action.squeeze(0), self.env.action_space.dtype)
        return out
    
    def learn(self, D, **kwargs):
        pass
    
    def checkpoint(self, logdir, num_iter):
        self.save(logdir/f'agent_{num_iter}.pth')
