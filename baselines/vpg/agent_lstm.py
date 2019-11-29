import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym.spaces as spaces
from lagom import BaseAgent
from lagom.utils import pickle_dump
from lagom.utils import numpify
from lagom.networks import Module
from lagom.networks import make_lnlstm
from lagom.networks import ortho_init
from lagom.networks import CategoricalHead
from lagom.networks import DiagGaussianHead
from lagom.networks import linear_lr_scheduler
from lagom.metric import bootstrapped_returns
from lagom.metric import gae
from lagom.transform import explained_variance as ev
from lagom.transform import describe


class FeatureNet(Module):
    def __init__(self, config, env, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        
        self.lstm = make_lnlstm(spaces.flatdim(env.observation_space), config['rnn.size'], num_layers=1)
        
    def forward(self, x, states):
        return self.lstm(x, states)


class Agent(BaseAgent):
    def __init__(self, config, env, **kwargs):
        super().__init__(config, env, **kwargs)
        
        feature_dim = config['rnn.size']
        self.feature_network = FeatureNet(config, env, **kwargs)
        if isinstance(env.action_space, spaces.Discrete):
            self.action_head = CategoricalHead(feature_dim, env.action_space.n, **kwargs)
        elif isinstance(env.action_space, spaces.Box):
            self.action_head = DiagGaussianHead(feature_dim, spaces.flatdim(env.action_space), config['agent.std0'], **kwargs)
        self.V_head = nn.Linear(feature_dim, 1)
        ortho_init(self.V_head, weight_scale=1.0, constant_bias=0.0)
        
        self.optimizer = optim.Adam(self.parameters(), lr=config['agent.lr'])
        if config['agent.use_lr_scheduler']:
            self.lr_scheduler = linear_lr_scheduler(self.optimizer, config['train.timestep'], min_lr=1e-8)

        self.register_buffer('total_timestep', torch.tensor(0))
        self.state = None
        
    def reset(self, batch_size):
        h = torch.zeros(batch_size, self.config['rnn.size']).to(self.config.device)
        c = torch.zeros_like(h)
        return h, c
        
    def choose_action(self, x, **kwargs):
        if x.first():
            self.state = self.reset(1)
        obs = torch.as_tensor(x.observation).float().to(self.config.device).unsqueeze(0)
        obs = obs.unsqueeze(0)  # add seq_dim
        features, [next_state] = self.feature_network(obs, [self.state])
        if 'last_info' not in kwargs:
            self.state = next_state
        features = features.squeeze(0)  # squeeze seq_dim
        action_dist = self.action_head(features)
        V = self.V_head(features)
        action = action_dist.sample()
        out = {}
        out['action_dist'] = action_dist
        out['V'] = V
        out['entropy'] = action_dist.entropy()
        out['action'] = action
        out['raw_action'] = numpify(action, self.env.action_space.dtype).squeeze(0)
        out['action_logprob'] = action_dist.log_prob(action.detach())
        return out
    
    def learn(self, D, **kwargs):
        logprobs = [torch.cat(traj.get_infos('action_logprob')) for traj in D]
        entropies = [torch.cat(traj.get_infos('entropy')) for traj in D]
        Vs = [torch.cat(traj.get_infos('V')) for traj in D]
        last_Vs = [traj.extra_info['last_info']['V'] for traj in D]
        Qs = [bootstrapped_returns(self.config['agent.gamma'], traj.rewards, last_V, traj.reach_terminal)
              for traj, last_V in zip(D, last_Vs)]
        As = [gae(self.config['agent.gamma'], self.config['agent.gae_lambda'], traj.rewards, V, last_V, traj.reach_terminal)
              for traj, V, last_V in zip(D, Vs, last_Vs)]
        
        # Metrics -> Tensor, device
        logprobs, entropies, Vs = map(lambda x: torch.cat(x).squeeze(), [logprobs, entropies, Vs])
        Qs, As = map(lambda x: torch.as_tensor(np.concatenate(x)).float().to(self.config.device), [Qs, As])
        if self.config['agent.standardize_adv']:
            As = (As - As.mean())/(As.std() + 1e-4)
        assert all([x.ndim == 1 for x in [logprobs, entropies, Vs, Qs, As]])
        
        # Loss
        policy_loss = -logprobs*As.detach()
        entropy_loss = -entropies
        value_loss = F.mse_loss(Vs, Qs, reduction='none')
        loss = policy_loss + self.config['agent.value_coef']*value_loss + self.config['agent.entropy_coef']*entropy_loss
        loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.config['agent.max_grad_norm'])
        self.optimizer.step()
        if self.config['agent.use_lr_scheduler']:
            self.lr_scheduler.step(self.total_timestep)
        self.total_timestep += sum([traj.T for traj in D])
        
        out = {}
        if self.config['agent.use_lr_scheduler']:
            out['current_lr'] = self.lr_scheduler.get_lr()
        out['loss'] = loss.item()
        out['grad_norm'] = grad_norm
        out['policy_loss'] = policy_loss.mean().item()
        out['entropy_loss'] = entropy_loss.mean().item()
        out['policy_entropy'] = -out['entropy_loss']
        out['value_loss'] = value_loss.mean().item()
        out['V'] = describe(numpify(Vs, 'float').squeeze(), axis=-1, repr_indent=1, repr_prefix='\n')
        out['explained_variance'] = ev(y_true=numpify(Qs, 'float'), y_pred=numpify(Vs, 'float'))
        return out
    
    def checkpoint(self, logdir, num_iter):
        self.save(logdir/f'agent_{num_iter}.pth')
        if self.config['env.normalize_obs']:
            moments = (self.env.obs_moments.mean, self.env.obs_moments.var)
            pickle_dump(obj=moments, f=logdir/f'obs_moments_{num_iter}', ext='.pth')
