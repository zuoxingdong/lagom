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
from lagom.policies import constraint_action

from lagom.value_functions import StateValueHead

from lagom.history.metrics import terminal_state_from_trajectory

from lagom.transform import Standardize

from lagom.agents import BaseAgent


class MLP(BaseNetwork):
    def make_params(self, config):
        self.feature_layers = make_fc(self.env_spec.observation_space.flat_dim, config['network.hidden_sizes'])
        
    def init_params(self, config):
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='tanh', constant_bias=0.0)
        
    def reset(self, config, **kwargs):
        pass
        
    def forward(self, x):
        for layer in self.feature_layers:
            x = torch.tanh(layer(x))
            
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
        self.V_head = StateValueHead(config, self.device, feature_dim)
    
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
        
        V = self.V_head(features)
        out['V'] = V
        
        if 'action_logprob' in out_keys:
            out['action_logprob'] = action_dist.log_prob(action)
        if 'entropy' in out_keys:
            out['entropy'] = action_dist.entropy()
        if 'perplexity' in out_keys:
            out['perplexity'] = action_dist.perplexity()
        
        return out
    

class Agent(BaseAgent):
    r"""Vanilla Policy Gradient (VPG) with value network (baseline), no bootstrapping to estimate value function. """
    def make_modules(self, config):
        self.policy = Policy(config, self.env_spec, self.device)
        
    def prepare(self, config, **kwargs):
        self.total_T = 0
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['algo.lr'])
        if config['algo.use_lr_scheduler']:
            if 'train.iter' in config:
                self.lr_scheduler = linear_lr_scheduler(self.optimizer, config['train.iter'], 'iteration-based')
            elif 'train.timestep' in config:
                self.lr_scheduler = linear_lr_scheduler(self.optimizer, config['train.timestep']+1, 'timestep-based')
        else:
            self.lr_scheduler = None

    def reset(self, config, **kwargs):
        pass

    def choose_action(self, obs, info={}):
        obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        
        if self.training:
            out = self.policy(obs, out_keys=['action', 'action_logprob', 'V', 'entropy'], info=info)
        else:
            with torch.no_grad():
                out = self.policy(obs, out_keys=['action'], info=info)
            
        # sanity check for NaN
        if torch.any(torch.isnan(out['action'])):
            while True:
                print('NaN !')
        if self.env_spec.control_type == 'Continuous':
            out['action'] = constraint_action(self.env_spec, out['action'])
            
        return out

    def learn(self, D, info={}):
        logprobs = [torch.stack(trajectory.all_info('action_logprob')) for trajectory in D]
        logprobs = torch.stack(logprobs)
        
        entropies = [torch.stack(trajectory.all_info('entropy')) for trajectory in D]
        entropies = torch.stack(entropies)
        
        Qs = [trajectory.all_discounted_returns(self.config['algo.gamma']) for trajectory in D]
        Qs = torch.tensor(Qs).float().to(self.device)
        if self.config['agent.standardize_Q']:
            Qs = (Qs - Qs.mean(1))/(Qs.std(1) + 1e-8)
            
        Vs = [torch.cat(trajectory.all_info('V')) for trajectory in D]
        Vs = torch.stack(Vs)
        
        As = Qs - Vs
        As = As.detach()
        if self.config['agent.standardize_adv']:
            As = (As - As.mean(1))/(As.std(1) + 1e-8)
        
        assert all([len(x.shape) == 2 for x in [logprobs, entropies, Qs, Vs]])
        
        policy_loss = -logprobs*As
        policy_loss = policy_loss.mean()
        entropy_loss = -entropies
        entropy_loss = entropy_loss.mean()
        value_loss = F.mse_loss(Vs, Qs, reduction='none')
        value_loss = value_loss.mean()
        
        entropy_coef = self.config['agent.entropy_coef']
        value_coef = self.config['agent.value_coef']
        loss = policy_loss + value_coef*value_loss + entropy_coef*entropy_loss
        
        if self.config['agent.fit_terminal_value']:
            terminal_states = [terminal_state_from_trajectory(trajectory) for trajectory in D if trajectory.complete]
            if len(terminal_states) > 0:
                terminal_states = torch.tensor(terminal_states).float().to(self.device)
                V_terminals = self.policy(terminal_states)['V']
            else:
                V_terminals = torch.tensor(0.0).to(self.device)
            
            V_terminals_loss = F.mse_loss(V_terminals, torch.zeros_like(V_terminals))
            V_terminals_coef = self.config['agent.V_terminals_coef']
            loss += V_terminals_coef*V_terminals_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config['agent.max_grad_norm'] is not None:
            clip_grad_norm_(self.parameters(), self.config['agent.max_grad_norm'])
        
        if self.lr_scheduler is not None:
            if self.lr_scheduler.mode == 'iteration-based':
                self.lr_scheduler.step()
            elif self.lr_scheduler.mode == 'timestep-based':
                self.lr_scheduler.step(self.total_T)
        
        self.optimizer.step()
        
        self.total_T += sum([trajectory.T for trajectory in D])
        
        out = {}
        out['loss'] = loss.item()
        out['policy_loss'] = policy_loss.item()
        out['entropy_loss'] = entropy_loss.item()
        out['value_loss'] = value_loss.item()
        if self.lr_scheduler is not None:
            out['current_lr'] = self.lr_scheduler.get_lr()

        return out
    
    @property
    def recurrent(self):
        pass
