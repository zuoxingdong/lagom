import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym.spaces as spaces
import lagom
import lagom.utils as utils
import lagom.rl as rl

from torch.utils.data import DataLoader
from baselines.ppo.dataset import Dataset


class Actor(lagom.nn.Module):
    def __init__(self, config, env, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        
        self.feature_layers = lagom.nn.make_fc(spaces.flatdim(env.observation_space), config['agent.policy.sizes'])
        for layer in self.feature_layers:
            lagom.nn.ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in config['agent.policy.sizes']])
        
        feature_dim = config['agent.policy.sizes'][-1]
        action_dim = spaces.flatdim(env.action_space)
        if isinstance(env.action_space, spaces.Discrete):
            self.action_head = lagom.nn.CategoricalHead(feature_dim, action_dim, **kwargs)
        elif isinstance(env.action_space, spaces.Box):
            kws = dict(std0=config['agent.std0'], min_var=config['agent.min_var'], max_var=config['agent.max_var'])
            self.action_head = lagom.nn.DiagGaussianHead(feature_dim, action_dim, **kws, **kwargs)

    def forward(self, x):
        for layer, layer_norm in zip(self.feature_layers, self.layer_norms):
            x = F.gelu(layer_norm(layer(x)))
        action_dist = self.action_head(x)
        return action_dist


class Critic(lagom.nn.Module):
    def __init__(self, config, env, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        
        self.feature_layers = lagom.nn.make_fc(spaces.flatdim(env.observation_space), config['agent.value.sizes'])
        for layer in self.feature_layers:
            lagom.nn.ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        
        feature_dim = config['agent.value.sizes'][-1]
        self.V_head = nn.Linear(feature_dim, 1)
        lagom.nn.ortho_init(self.V_head, weight_scale=1.0, constant_bias=0.0)

    def forward(self, x):
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        V = self.V_head(x)
        return V


class Agent(lagom.BaseAgent):
    def __init__(self, config, env, **kwargs):
        super().__init__(config, env, **kwargs)
        
        self.policy = Actor(config, env, **kwargs)
        self.value = Critic(config, env, **kwargs)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config['agent.policy.lr'])
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=config['agent.value.lr'])
        
        self.register_buffer('total_timestep', torch.tensor(0))
        
    def choose_action(self, x, **kwargs):
        obs = torch.as_tensor(x.observation).float().to(self.config.device).unsqueeze(0)
        action_dist = self.policy(obs)
        V = self.value(obs)
        action = action_dist.sample()
        out = {}
        out['action_dist'] = action_dist
        out['V'] = V
        out['entropy'] = action_dist.entropy()
        out['action'] = action
        out['raw_action'] = utils.numpify(action.squeeze(0), self.env.action_space.dtype)
        out['action_logprob'] = action_dist.log_prob(action.detach())
        return out
    
    def learn_one_update(self, data):
        data = [d.to(self.config.device) for d in data]
        observations, old_actions, old_logprobs, old_entropies, old_Vs, old_Qs, old_As = data
        
        action_dist = self.policy(observations)
        logprobs = action_dist.log_prob(old_actions).squeeze()
        entropies = action_dist.entropy().squeeze()
        Vs = self.value(observations).squeeze()
        assert all([x.ndim == 1 for x in [logprobs, entropies, Vs]])
        
        ratio = torch.exp(logprobs - old_logprobs)
        eps = self.config['agent.clip_range']
        policy_loss = -torch.min(ratio*old_As, 
                                 torch.clamp(ratio, 1.0 - eps, 1.0 + eps)*old_As)
        policy_loss = policy_loss.mean(0)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['agent.max_grad_norm'])
        self.policy_optimizer.step()
        
        clipped_Vs = old_Vs + torch.clamp(Vs - old_Vs, -eps, eps)
        value_loss = torch.max(F.mse_loss(Vs, old_Qs, reduction='none'), 
                               F.mse_loss(clipped_Vs, old_Qs, reduction='none'))
        value_loss = value_loss.mean(0)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        value_grad_norm = nn.utils.clip_grad_norm_(self.value.parameters(), self.config['agent.max_grad_norm'])
        self.value_optimizer.step()
        
        out = {}
        out['policy_grad_norm'] = policy_grad_norm
        out['value_grad_norm'] = value_grad_norm
        out['policy_loss'] = policy_loss.item()
        out['policy_entropy'] = entropies.mean().item()
        out['value_loss'] = value_loss.item()
        out['explained_variance'] = utils.explained_variance(y_true=old_Qs.tolist(), y_pred=Vs.tolist())
        out['approx_kl'] = (old_logprobs - logprobs).mean(0).item()
        out['clip_frac'] = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).float().mean(0).item()
        return out
        
    def learn(self, D, **kwargs):
        get_info = lambda key: [torch.cat(traj.get_infos(key)) for traj in D]
        logprobs, entropies, Vs = map(get_info, ['action_logprob', 'entropy', 'V'])
        last_Vs = [traj.extra_info['last_info']['V'] for traj in D]
        Qs = [rl.bootstrapped_returns(self.config['agent.gamma'], traj.rewards, last_V, traj.reach_terminal)
              for traj, last_V in zip(D, last_Vs)]
        As = [rl.gae(self.config['agent.gamma'], self.config['agent.gae_lambda'], traj.rewards, V, last_V, traj.reach_terminal)
              for traj, V, last_V in zip(D, Vs, last_Vs)]
        
        # Handle dtype, device
        logprobs, entropies, Vs = map(lambda x: torch.cat(x).squeeze(), [logprobs, entropies, Vs])
        Qs, As = map(lambda x: torch.as_tensor(np.concatenate(x)).float().squeeze().to(self.config.device), [Qs, As])
        if self.config['agent.standardize_adv']:
            As = (As - As.mean())/(As.std() + 1e-4)
        assert all([x.ndim == 1 for x in [logprobs, entropies, Vs, Qs, As]])
        
        dataset = Dataset(D, logprobs, entropies, Vs, Qs, As)
        dataloader = DataLoader(dataset, self.config['train.batch_size'], shuffle=True)
        for epoch in range(self.config['train.num_epochs']):
            logs = [self.learn_one_update(data) for data in dataloader]

        self.total_timestep += sum([traj.T for traj in D])
        out = {}
        out['policy_grad_norm'] = np.mean([item['policy_grad_norm'] for item in logs])
        out['value_grad_norm'] = np.mean([item['value_grad_norm'] for item in logs])
        out['policy_loss'] = np.mean([item['policy_loss'] for item in logs])
        out['policy_entropy'] = np.mean([item['policy_entropy'] for item in logs])
        out['value_loss'] = np.mean([item['value_loss'] for item in logs])
        out['explained_variance'] = np.mean([item['explained_variance'] for item in logs])
        out['approx_kl'] = np.mean([item['approx_kl'] for item in logs])
        out['clip_frac'] = np.mean([item['clip_frac'] for item in logs])
        return out
    
    def checkpoint(self, logdir, num_iter):
        self.save(logdir/f'agent_{num_iter}.pth')
        if self.config['env.normalize_obs']:
            moments = (self.env.obs_moments.mean, self.env.obs_moments.var)
            utils.pickle_dump(obj=moments, f=logdir/f'obs_moments_{num_iter}', ext='.pth')
