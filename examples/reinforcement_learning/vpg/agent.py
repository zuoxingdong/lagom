import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_

from lagom.networks import BaseNetwork
from lagom.networks import BaseRNN
from lagom.networks import make_fc
from lagom.networks import make_rnncell
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


class NN(BaseNetwork):
    def make_params(self, config):
        self.feature_layers = make_fc(self.env_spec.observation_space.flat_dim, config['network.hidden_sizes'])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in config['network.hidden_sizes']])
        
    def init_params(self, config):
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='leaky_relu', constant_bias=0.0)

    @property
    def recurrent(self):
        return False
            
    def reset(self, config, **kwargs):
        pass
        
    def forward(self, x):
        for layer, layer_norm in zip(self.feature_layers, self.layer_norms):
            x = layer_norm(F.celu(layer(x)))
            
        return x
    
    
class Critic(NN):
    def make_params(self, config):
        super().make_params(config)
        self.output_layer = StateValueHead(config, self.device, config['network.hidden_sizes'][-1])
        
    def init_params(self, config):
        super().init_params(config)
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
        
    @property
    def recurrent(self):
        return super().recurrent
    
    def reset(self, config, **kwargs):
        super().reset(config, **kwargs)
        
    def forward(self, x, **kwargs):
        x = super().forward(x)
        x = self.output_layer(x)
            
        out = {'V': x}
            
        return out
    
    
class RNN(BaseRNN):
    def make_params(self, config):
        self.rnn = nn.LSTM(self.env_spec.observation_space.flat_dim, 
                           config['network.hidden_sizes'][-1], 
                           num_layers=1)

    def init_params(self, config):
        ortho_init(self.rnn, weight_scale=1.0, constant_bias=0.0)

    @property
    def recurrent(self):
        return True
            
    def reset(self, config, **kwargs):
        self.h, self.c = self.init_hidden_states(config, config['train.N'], **kwargs)
        
    def init_hidden_states(self, config, batch_size, **kwargs):
        h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(self.device)
        c = torch.zeros_like(h)
        
        return h, c

    def forward(self, x, hidden_states, mask=None, **kwargs):
        # augment seq_len dimension 1 to input: [seq_len, batch_size, input_size]
        x = x.unsqueeze(0)
        
        h, c = hidden_states
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(-1).to(self.device)
            h *= mask
            c *= mask
        
        output, (h, c) = self.rnn(x, (h, c))
        output = output.squeeze(0)  # remove seq_len dim
        
        return output, (h, c)
    
    
class RNNCritic(RNN):
    def make_params(self, config):
        super().make_params(config)
        self.output_layer = StateValueHead(config, self.device, self.rnn.hidden_size)
        
    def init_params(self, config):
        super().init_params(config)
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
        
    @property
    def recurrent(self):
        return super().recurrent
    
    def reset(self, config, **kwargs):
        super().reset(config, **kwargs)
        
    def forward(self, x, **kwargs):
        out = {}
        
        output, (h, c) = super().forward(x, (self.h, self.c), **kwargs)
        self.h, self.c = h, c
        
        out['V'] = self.output_layer(output)
        out['V_rnn_states'] = (h, c)
        
        return out
    
    
class Policy(BasePolicy):
    def make_networks(self, config):
        if config['network.recurrent']:
            self.feature_network = RNN(config, self.device, env_spec=self.env_spec)
        else:
            self.feature_network = NN(config, self.device, env_spec=self.env_spec)
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
        self.optimizer = optim.Adam(self.parameters(), lr=config['algo.lr'])
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
        return self.feature_network.recurrent
    
    def reset(self, config, **kwargs):
        self.feature_network.reset(config, **kwargs)

    def __call__(self, x, out_keys=['action', 'V'], **kwargs):
        out = {}
        
        if self.feature_network.recurrent:
            features, (h, c) = self.feature_network(x, (self.feature_network.h, self.feature_network.c), **kwargs)
            self.feature_network.h, self.feature_network.c = h, c
            out['rnn_states'] = (h, c)
        else:
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
    r"""Vanilla Policy Gradient (VPG) with value network (baseline) and GAE."""
    def make_modules(self, config):
        self.policy = Policy(config, self.env_spec, self.device)
        if config['network.independent_V']:
            if config['network.recurrent']:
                self.critic = RNNCritic(config, self.device, env_spec=self.env_spec)
            else:
                self.critic = Critic(config, self.device, env_spec=self.env_spec)
        
    def prepare(self, config, **kwargs):
        self.total_T = 0

    def reset(self, config, **kwargs):
        self.policy.reset(config, **kwargs)
        if config['network.independent_V']:
            self.critic.reset(config, **kwargs)

    def choose_action(self, obs, **kwargs):
        obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        
        if self.training:
            if self.config['network.independent_V']:
                out = self.policy(obs, out_keys=['action', 'action_logprob', 'entropy'], **kwargs)
                V_out = self.critic(obs, **kwargs)
                out = {**out, **V_out}
            else:
                out = self.policy(obs, out_keys=['action', 'action_logprob', 'V', 'entropy'], **kwargs)
        else:
            with torch.no_grad():
                out = self.policy(obs, out_keys=['action'], **kwargs)
            
        # sanity check for NaN
        if torch.any(torch.isnan(out['action'])):
            raise ValueError('NaN!')
            
        return out

    def learn(self, D, **kwargs):
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
                last_Vs = self.critic(last_states)['V']
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
        
        policy_loss = -logprobs*As
        policy_loss = policy_loss.mean()
        entropy_loss = -entropies
        entropy_loss = entropy_loss.mean()
        value_loss = F.mse_loss(all_Vs, Qs, reduction='none')
        value_loss = value_loss.mean()
        
        entropy_coef = self.config['agent.entropy_coef']
        value_coef = self.config['agent.value_coef']
        loss = policy_loss + value_coef*value_loss + entropy_coef*entropy_loss
        
        if self.config['agent.fit_terminal_value']:
            terminal_states = terminal_state_from_episode(D)
            if terminal_states is not None:
                terminal_states = torch.from_numpy(terminal_states).float().to(self.device)
                terminal_Vs = self.policy(terminal_states, out_keys=['V'])['V']
                terminal_value_loss = F.mse_loss(terminal_Vs, torch.zeros_like(terminal_Vs))
                terminal_value_loss_coef = self.config['agent.terminal_value_coef']
                loss += terminal_value_loss_coef*terminal_value_loss
        
        self.policy.optimizer.zero_grad()
        if self.config['network.independent_V']:
            self.critic.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer_step(self.config, total_T=self.total_T)
        if self.config['network.independent_V']:
            self.critic.optimizer_step(self.config, total_T=self.total_T)
        
        self.total_T += D.total_T
        
        out = {}
        if self.policy.lr_scheduler is not None:
            out['current_lr'] = self.policy.lr_scheduler.get_lr()
        out['loss'] = loss.item()
        out['policy_loss'] = policy_loss.item()
        out['entropy_loss'] = entropy_loss.item()
        out['policy_entropy'] = -entropy_loss.item()
        out['value_loss'] = value_loss.item()
        ev = ExplainedVariance()
        ev = ev(y_true=Qs.detach().cpu().numpy().squeeze(), y_pred=all_Vs.detach().cpu().numpy().squeeze())
        out['explained_variance'] = ev
        
        return out

    @property
    def recurrent(self):
        return self.policy.recurrent
