import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym.spaces as spaces
import lagom
import lagom.utils as utils
import lagom.rl as rl


class MLP(lagom.nn.Module):
    def __init__(self, config, env, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        
        self.feature_layers = lagom.nn.make_fc(spaces.flatdim(env.observation_space), config['nn.sizes'])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in config['nn.sizes']])
        
    def forward(self, x):
        for layer, layer_norm in zip(self.feature_layers, self.layer_norms):
            x = layer_norm(F.relu(layer(x)))
        return x


class Agent(lagom.BaseAgent):
    def __init__(self, config, env, **kwargs):
        super().__init__(config, env, **kwargs)
        
        self.feature_network = MLP(config, env, **kwargs)
        feature_dim = config['nn.sizes'][-1]
        if isinstance(env.action_space, spaces.Discrete):
            self.action_head = lagom.nn.CategoricalHead(feature_dim, env.action_space.n, **kwargs)
        elif isinstance(env.action_space, spaces.Box):
            self.action_head = lagom.nn.DiagGaussianHead(feature_dim, spaces.flatdim(env.action_space), 0.5, **kwargs)

        self.register_buffer('total_timestep', torch.tensor(0))
        
    def choose_action(self, x, **kwargs):
        obs = torch.as_tensor(x.observation).float().to(self.config.device).unsqueeze(0)
        features = self.feature_network(obs)
        action_dist = self.action_head(features)
        action = action_dist.sample()
        out = {}
        out['raw_action'] = utils.numpify(action.squeeze(0), self.env.action_space.dtype)
        return out
    
    def learn(self, D, **kwargs):
        pass
    
    def checkpoint(self, logdir, num_iter):
        self.save(logdir/f'agent_{num_iter}.pth')
        if 'env.normalize_obs' in self.config and self.config['env.normalize_obs']:
            moments = (self.env.obs_moments.mean, self.env.obs_moments.var)
            utils.pickle_dump(obj=moments, f=logdir/f'obs_moments_{num_iter}', ext='.pth')
