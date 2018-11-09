import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_

from torch.utils.data import DataLoader

from lagom.networks import BaseNetwork
from lagom.networks import make_fc
from lagom.networks import ortho_init
from lagom.networks import linear_lr_scheduler

from lagom.policies import BasePolicy
from lagom.policies import CategoricalHead
from lagom.policies import DiagGaussianHead
from lagom.policies import constraint_action

from lagom.value_functions import StateValueHead

from lagom.transform import Standardize
from lagom.transform import ExplainedVariance

from lagom.agents import BaseAgent

from dataset import Dataset


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
    r"""Proximal policy optimization (PPO). """
    def make_modules(self, config):
        self.policy = Policy(config, self.env_spec, self.device)
        
    def prepare(self, config, **kwargs):
        self.total_T = 0
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['algo.lr'], eps=1e-5)
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
    
    def learn_one_update(self, data):
        data = [d.to(self.device) for d in data]
        states, old_logprobs, As, old_Vs, Qs = data
        
        out = self.policy(states, out_keys=['action', 'action_logprob', 'V', 'entropy'])
        logprobs = out['action_logprob']
        Vs = out['V'].squeeze(1)
        entropies = out['entropy']
        
        if self.config['agent.standardize_adv']:
            As = (As - As.mean())/(As.std() + 1e-8)
        ratio = torch.exp(logprobs - old_logprobs)
        eps = self.config['agent.clip_range']
        policy_loss = torch.min(ratio*As, torch.clamp(ratio, 1.0 - eps, 1.0 + eps)*As)
        policy_loss = -policy_loss.mean()
        
        if self.config['agent.standardize_Q']:
            Qs = (Qs - Qs.mean())/(Qs.std() + 1e-8)
        clipped_Vs = old_Vs + torch.clamp(Vs - old_Vs, -eps, eps)
        value_loss = torch.max(F.mse_loss(Vs, Qs, reduction='none'), F.mse_loss(clipped_Vs, Qs, reduction='none'))
        value_loss = value_loss.mean()
        
        entropy_loss = -entropies
        entropy_loss = entropy_loss.mean()
        
        entropy_coef = self.config['agent.entropy_coef']
        value_coef = self.config['agent.value_coef']
        loss = policy_loss + value_coef*value_loss + entropy_coef*entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config['agent.max_grad_norm'] is not None:
            clip_grad_norm_(self.parameters(), self.config['agent.max_grad_norm'])
        
        self.optimizer.step()
        
        out = {}
        out['loss'] = loss.item()
        out['policy_loss'] = policy_loss.item()
        out['value_loss'] = value_loss.item()
        out['entropy_loss'] = entropy_loss.item()
        ev = ExplainedVariance()(y_true=Qs.cpu().detach().numpy(), y_pred=Vs.cpu().detach().numpy())
        out['explained_variance'] = ev
        
        return out
        
    def learn(self, D, info={}):
        dataset = Dataset(self.config, D, self.policy)
        if self.config['cuda']:
            kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            kwargs = {}
        dataloader = DataLoader(dataset, self.config['train.batch_size'], shuffle=True, **kwargs)
        
        if self.lr_scheduler is not None:
            if self.lr_scheduler.mode == 'iteration-based':
                self.lr_scheduler.step()
            elif self.lr_scheduler.mode == 'timestep-based':
                self.lr_scheduler.step(self.total_T)
        
        for epoch in range(self.config['train.num_epochs']):
            loss = []
            policy_loss = []
            value_loss = []
            entropy_loss = []
            explained_variance = []
            for data in dataloader:
                out = self.learn_one_update(data)
                
                loss.append(out['loss'])
                policy_loss.append(out['policy_loss'])
                value_loss.append(out['value_loss'])
                entropy_loss.append(out['entropy_loss'])
                explained_variance.append(out['explained_variance'])
        
        loss = np.mean(loss)
        policy_loss = np.mean(policy_loss)
        value_loss = np.mean(value_loss)
        entropy_loss = np.mean(entropy_loss)
        explained_variance = np.mean(explained_variance)
        
        self.total_T += sum([segment.T for segment in D])
        
        out = {}
        out['loss'] = loss
        out['policy_loss'] = policy_loss
        out['value_loss'] = value_loss
        out['entropy_loss'] = entropy_loss
        out['explained_variance'] = explained_variance
        if self.lr_scheduler is not None:
            out['current_lr'] = self.lr_scheduler.get_lr()
            
        return out
        
    @property
    def recurrent(self):
        pass
