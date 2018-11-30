import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_

from lagom.networks import BaseNetwork
from lagom.networks import make_fc
from lagom.networks import ortho_init
from lagom.networks import linear_lr_scheduler

from lagom.policies import BasePolicy
from lagom.policies import CategoricalHead
from lagom.policies import DiagGaussianHead

from lagom.value_functions import StateValueHead

from lagom.transform import ExplainedVariance

from lagom.history.metrics import final_state_from_episode
from lagom.history.metrics import terminal_state_from_episode
from lagom.history.metrics import bootstrapped_returns_from_episode
from lagom.history.metrics import gae_from_episode

from lagom.agents import BaseAgent

from torch.utils.data import DataLoader
from dataset import Dataset


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
    
    
class Critic(BaseNetwork):
    def make_params(self, config):
        self.feature_layers = make_fc(self.env_spec.observation_space.flat_dim, config['network.hidden_sizes'])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in config['network.hidden_sizes']])
        self.output_layer = StateValueHead(config, self.device, config['network.hidden_sizes'][-1])
        
    def init_params(self, config):
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='leaky_relu', constant_bias=0.0)
            
        self.make_optimizer(config)

    def make_optimizer(self, config, **kwargs):
        self.optimizer = optim.Adam(self.parameters(), lr=config['algo.lr_V'])
        if config['algo.use_lr_scheduler']:
            if 'train.iter' in config:
                self.lr_scheduler = linear_lr_scheduler(self.optimizer, config['train.iter'], 'iteration-based')
            elif 'train.timestep' in config:
                self.lr_scheduler = linear_lr_scheduler(self.optimizer, config['train.timestep']+1, 'timestep-based')
        else:
            self.lr_scheduler = None
            
    def optimizer_step(self, config, **kwargs):
        if config['agent.max_grad_norm'] is not None:
            clip_grad_norm_(self.parameters(), config['agent.max_grad_norm'])
        
        if self.lr_scheduler is not None:
            if self.lr_scheduler.mode == 'iteration-based':
                self.lr_scheduler.step()
            elif self.lr_scheduler.mode == 'timestep-based':
                self.lr_scheduler.step(kwargs['total_T'])
        
        self.optimizer.step()
            
    def reset(self, config, **kwargs):
        pass
        
    def forward(self, x):
        for layer, layer_norm in zip(self.feature_layers, self.layer_norms):
            x = layer_norm(F.celu(layer(x)))
        x = self.output_layer(x)
            
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
                                                min_std=config['agent.min_std'], 
                                                std_style=config['agent.std_style'], 
                                                constant_std=config['agent.constant_std'],
                                                std_state_dependent=config['agent.std_state_dependent'],
                                                init_std=config['agent.init_std'])
        if not config['network.independent_V']:
            self.V_head = StateValueHead(config, self.device, feature_dim)
    
    def make_optimizer(self, config, **kwargs):
        self.optimizer = optim.Adam(self.parameters(), lr=config['algo.lr'])###################################################################, eps=1e-5)
        if config['algo.use_lr_scheduler']:
            if 'train.iter' in config:
                self.lr_scheduler = linear_lr_scheduler(self.optimizer, config['train.iter'], 'iteration-based')
            elif 'train.timestep' in config:
                self.lr_scheduler = linear_lr_scheduler(self.optimizer, config['train.timestep']+1, 'timestep-based')
        else:
            self.lr_scheduler = None
            
    def optimizer_step(self, config, **kwargs):
        if config['agent.max_grad_norm'] is not None:
            clip_grad_norm_(self.parameters(), config['agent.max_grad_norm'])
        
        if self.lr_scheduler is not None:
            if self.lr_scheduler.mode == 'iteration-based':
                self.lr_scheduler.step()
            elif self.lr_scheduler.mode == 'timestep-based':
                self.lr_scheduler.step(kwargs['total_T'])
        
        self.optimizer.step()
    
    @property
    def recurrent(self):
        return False
    
    def reset(self, config, **kwargs):
        pass

    def __call__(self, x, out_keys=['action', 'V'], info={}, **kwargs):
        out = {}
        
        features = self.feature_network(x)
        action_dist = self.action_head(features)
        
        action = action_dist.sample().detach()  # TODO: detach is necessary or not ?
        out['action'] = action
        
        if 'V' in out_keys:
            V = self.V_head(features)
            out['V'] = V
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
    r"""Proximal policy optimization (PPO). """
    def make_modules(self, config):
        self.policy = Policy(config, self.env_spec, self.device)
        if config['network.independent_V']:
            self.critic = Critic(config, self.device, env_spec=self.env_spec)
        
    def prepare(self, config, **kwargs):
        self.total_T = 0

    def reset(self, config, **kwargs):
        pass

    def choose_action(self, obs, info={}):
        obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        
        if self.training:
            if self.config['network.independent_V']:
                out = self.policy(obs, out_keys=['action', 'action_logprob', 'entropy'], info=info)
                out['V'] = self.critic(obs)
            else:
                out = self.policy(obs, out_keys=['action', 'action_logprob', 'V', 'entropy'], info=info)
        else:
            with torch.no_grad():
                out = self.policy(obs, out_keys=['action'], info=info)
            
        # sanity check for NaN
        if torch.any(torch.isnan(out['action'])):
            raise ValueError('NaN!')
            
        return out
    
    def learn_one_update(self, data):
        data = [d.to(self.device) for d in data]
        observations, old_actions, old_logprobs, old_entropies, old_all_Vs, old_Qs, old_As = data
        
        if self.config['network.independent_V']:
            out = self.policy(observations, out_keys=['action_dist', 'entropy'])
            out['V'] = self.critic(observations)
        else:
            out = self.policy(observations, out_keys=['action_dist', 'V', 'entropy'])
        action_dist = out['action_dist']
        logprobs = action_dist.log_prob(old_actions).squeeze(-1)
        entropies = out['entropy'].squeeze(-1)
        all_Vs = out['V'].squeeze(-1)
        
        ratio = torch.exp(logprobs - old_logprobs)
        eps = self.config['agent.clip_range']
        policy_loss = -torch.min(ratio*old_As, 
                                 torch.clamp(ratio, 1.0 - eps, 1.0 + eps)*old_As)
        policy_loss = policy_loss.mean()
        entropy_loss = -entropies
        entropy_loss = entropy_loss.mean()
        clipped_all_Vs = old_all_Vs + torch.clamp(all_Vs - old_all_Vs, -eps, eps)
        value_loss = torch.max(F.mse_loss(all_Vs, old_Qs, reduction='none'), 
                               F.mse_loss(clipped_all_Vs, old_Qs, reduction='none'))
        value_loss = value_loss.mean()
        
        entropy_coef = self.config['agent.entropy_coef']
        value_coef = self.config['agent.value_coef']
        loss = policy_loss + value_coef*value_loss + entropy_coef*entropy_loss
        
        self.policy.optimizer.zero_grad()
        if self.config['network.independent_V']:
            self.critic.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer_step(self.config, total_T=self.total_T)
        if self.config['network.independent_V']:
            self.critic.optimizer_step(self.config, total_T=self.total_T)
        
        approx_kl = torch.mean(old_logprobs - logprobs)
        clip_frac = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).float().mean()
        
        out = {}
        out['loss'] = loss.item()
        out['policy_loss'] = policy_loss.item()
        out['entropy_loss'] = entropy_loss.item()
        out['value_loss'] = value_loss.item()
        ev = ExplainedVariance()
        ev = ev(y_true=old_Qs.detach().cpu().numpy().squeeze(), y_pred=all_Vs.detach().cpu().numpy().squeeze())
        out['explained_variance'] = ev
        out['approx_kl'] = approx_kl.item()
        out['clip_frac'] = clip_frac.item()

        return out
        
    def learn(self, D, info={}):
        # mask-out values when its environment already terminated for episodes
        episode_validity_masks = torch.from_numpy(D.numpy_validity_masks).float().to(self.device)
        logprobs = torch.stack([info['action_logprob'] for info in D.batch_infos], 1).squeeze(-1)
        logprobs = logprobs*episode_validity_masks
        entropies = torch.stack([info['entropy'] for info in D.batch_infos], 1).squeeze(-1)
        entropies = entropies*episode_validity_masks
        all_Vs = torch.stack([info['V'] for info in D.batch_infos], 1).squeeze(-1)
        all_Vs = all_Vs*episode_validity_masks
        
        last_states = torch.from_numpy(final_state_from_episode(D)).float().to(self.device)
        with torch.no_grad():
            if self.config['network.independent_V']:
                last_Vs = self.critic(last_states)
            else:
                last_Vs = self.policy(last_states, out_keys=['V'])['V']
        Qs = bootstrapped_returns_from_episode(D, last_Vs, self.config['algo.gamma'])
        Qs = torch.from_numpy(Qs.copy()).float().to(self.device)
        if self.config['agent.standardize_Q']:
            Qs = (Qs - Qs.mean(1, keepdim=True))/(Qs.std(1, keepdim=True) + 1e-8)
        
        As = gae_from_episode(D, all_Vs, last_Vs, self.config['algo.gamma'], self.config['algo.gae_lambda'])
        As = torch.from_numpy(As.copy()).float().to(self.device)
        if self.config['agent.standardize_adv']:
            As = (As - As.mean(1, keepdim=True))/(As.std(1, keepdim=True) + 1e-8)
        
        assert all([x.ndimension() == 2 for x in [logprobs, entropies, all_Vs, Qs, As]])
        
        dataset = Dataset(self.env_spec, D.numpy_observations, D.numpy_actions, logprobs, entropies, all_Vs, Qs, As)
        if self.config['cuda']:
            kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            kwargs = {}
        dataloader = DataLoader(dataset, self.config['train.batch_size'], shuffle=True, **kwargs)
        
        for epoch in range(self.config['train.num_epochs']):
            out = [self.learn_one_update(data) for data in dataloader]
            
            loss = np.mean([item['loss'] for item in out])
            policy_loss = np.mean([item['policy_loss'] for item in out])
            entropy_loss = np.mean([item['entropy_loss'] for item in out])
            value_loss = np.mean([item['value_loss'] for item in out])
            explained_variance = np.mean([item['explained_variance'] for item in out])
            approx_kl = np.mean([item['approx_kl'] for item in out])
            clip_frac = np.mean([item['clip_frac'] for item in out])
            
            if approx_kl > self.config['agent.target_kl']:
                break
        
        self.total_T += D.total_T
        
        out = {}
        if self.policy.lr_scheduler is not None:
            out['current_lr'] = self.policy.lr_scheduler.get_lr()
        out['loss'] = loss
        out['policy_loss'] = policy_loss
        out['entropy_loss'] = entropy_loss
        out['policy_entropy'] = -entropy_loss
        out['value_loss'] = value_loss
        out['explained_variance'] = explained_variance
        out['approx_kl'] = approx_kl
        out['clip_frac'] = clip_frac
        out['finished_inner_epochs'] = f'{epoch+1}/{self.config["train.num_epochs"]}'
        
        return out

    @property
    def recurrent(self):
        pass
