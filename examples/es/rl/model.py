import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.networks import BaseNetwork
from lagom.networks import make_fc
from lagom.networks import ortho_init

from lagom.policies import BasePolicy
from lagom.policies import CategoricalHead
from lagom.policies import DiagGaussianHead

from lagom.agents import BaseAgent


class MLP(BaseNetwork):
    def make_params(self, config):
        self.feature_layers = make_fc(self.env_spec.observation_space.flat_dim, config['network.hidden_sizes'])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in config['network.hidden_sizes']])
        
    def init_params(self, config):
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='leaky_relu', constant_bias=0.0)

    def reset(self, config, **kwargs):
        pass
        
    def forward(self, x):
        for layer, layer_norm in zip(self.feature_layers, self.layer_norms):
            x = layer_norm(F.celu(layer(x)))
            
        return x
    
    
class Policy(BasePolicy):
    def make_networks(self, config):
        self.feature_network = MLP(config, self.device, env_spec=self.env_spec)
        feature_dim = config['network.hidden_sizes'][-1]
        
        if self.env_spec.control_type == 'Discrete':
            self.action_head = CategoricalHead(config, self.device, feature_dim, self.env_spec)
        elif self.env_spec.control_type == 'Continuous':
            self.action_head = DiagGaussianHead(config, 
                                                self.device, 
                                                feature_dim, 
                                                self.env_spec, 
                                                min_std=1e-06, 
                                                std_style='exp', 
                                                constant_std=None, 
                                                std_state_dependent=False, 
                                                init_std=0.5)
    
    def make_optimizer(self, config, **kwargs):
        pass
    
    def optimizer_step(self, config, **kwargs):
        pass
    
    @property
    def recurrent(self):
        return False
    
    def reset(self, config, **kwargs):
        pass

    def __call__(self, x, out_keys=['action'], info={}, **kwargs):
        out = {}
        
        features = self.feature_network(x)
        action_dist = self.action_head(features)
        
        action = action_dist.sample().detach()
        out['action'] = action
        
        if 'action_dist' in out_keys:
            out['action_dist'] = action_dist
        if 'action_logprob' in out_keys:
            out['action_logprob'] = action_dist.log_prob(action)
        if 'entropy' in out_keys:
            out['entropy'] = action_dist.entropy()
        if 'perplexity' in out_keys:
            out['perplexity'] = action_dist.perplexity()
        
        return out
    

class Agent(BaseAgent):
    def make_modules(self, config):
        self.policy = Policy(config, self.env_spec, self.device)
        
    def prepare(self, config, **kwargs):
        self.total_T = 0

    def reset(self, config, **kwargs):
        pass

    def choose_action(self, obs, info={}):
        obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        
        with torch.no_grad():
            out = self.policy(obs, out_keys=['action'], info=info)
            
        # sanity check for NaN
        if torch.any(torch.isnan(out['action'])):
            raise ValueError('NaN!')

        return out

    def learn(self, D, info={}):
        pass

    @property
    def recurrent(self):
        pass
