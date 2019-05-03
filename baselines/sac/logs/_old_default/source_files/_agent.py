import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Independent
from torch.distributions import Normal
from torch.distributions import Transform
from torch.distributions import TransformedDistribution
from torch.distributions import constraints

from lagom import BaseAgent
from lagom.transform import describe
from lagom.utils import pickle_dump
from lagom.envs import flatdim
from lagom.networks import Module
from lagom.networks import make_fc
from lagom.networks import ortho_init

  

class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5*(x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        eps = torch.finfo(y.dtype).eps
        return self.atanh(y.clamp(min=-1. + eps, max=1. +- eps))

    def log_abs_det_jacobian(self, x, y):
        return 2.*(np.log(2.) - x - F.softplus(-2.*x))
    
        
### Use with NormalizeAction wrapper together
class TanhDiagGaussianHead(Module):
    r"""Defines a module for a tanh-squashed diagonal Gaussian (continuous) action distribution
    which the standard deviation is dependent on the state. 
    
    This is particularly useful for SAC, because it maximizes trade-off between reward and entropy.
    Entropy must be unique to state. For ReLU network, a randomly initialized network can produce 
    very large value for logstd, which results in either entirely deterministic or too random
    to come back to earth. Either of these introduces numerical instability which could break 
    the algorithm. We constraint logstd between a range. 
    
    """
    def __init__(self, feature_dim, action_dim, device, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.device = device
        
        self.logstd_min = -20
        self.logstd_max = 2
        
        self.mean_head = nn.Linear(self.feature_dim, self.action_dim)
        # 0.01 -> almost zeros initially
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        
        self.logstd_head = nn.Linear(self.feature_dim, self.action_dim)
        
        self.to(self.device)
        
    def forward(self, x):
        mean = self.mean_head(x)
        logstd = torch.tanh(self.logstd_head(x))
        logstd = self.logstd_min + 0.5*(1 + logstd)*(self.logstd_max - self.logstd_min)
        std = torch.exp(logstd)
        action_dist = TransformedDistribution(Independent(Normal(loc=mean, scale=std), 1), [TanhTransform()])
        return action_dist
        
    def mean_forward(self, x):
        mean = self.mean_head(x)
        return torch.tanh(mean)
    
        
class Actor(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        self.feature_layers = make_fc(flatdim(env.observation_space), [256, 256])
        self.action_head = TanhDiagGaussianHead(256, flatdim(env.action_space), device, **kwargs)
        
        self.to(device)

    def forward(self, x):
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        action_dist = self.action_head(x)
        return action_dist
    
    def mean_forward(self, x):
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        action = self.action_head.mean_forward(x)
        return action


class Critic(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        # Q1
        self.first_feature_layers = make_fc(flatdim(env.observation_space) + flatdim(env.action_space), [256, 256])
        self.first_Q_head = nn.Linear(256, 1)
        
        # Q2
        self.second_feature_layers = make_fc(flatdim(env.observation_space) + flatdim(env.action_space), [256, 256])
        self.second_Q_head = nn.Linear(256, 1)
        
        self.to(self.device)
        
    def Q1(self, x, action):
        x = torch.cat([x, action], dim=-1)
        for layer in self.first_feature_layers:
            x = F.relu(layer(x))
        x = self.first_Q_head(x)
        return x
    
    def Q2(self, x, action):
        x = torch.cat([x, action], dim=-1)
        for layer in self.second_feature_layers:
            x = F.relu(layer(x))
        x = self.second_Q_head(x)
        return x
        
    def forward(self, x, action):
        return self.Q1(x, action), self.Q2(x, action)


class Agent(BaseAgent):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(config, env, device, **kwargs)
        
        self.actor = Actor(config, env, device, **kwargs)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['agent.actor.lr'])
        
        self.critic = Critic(config, env, device, **kwargs)
        self.critic_target = Critic(config, env, device, **kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['agent.critic.lr'])
        
        self.target_entropy = -float(flatdim(env.action_space))
        self.log_alpha = nn.Parameter(torch.tensor(np.log(config['agent.alpha0'])).to(device))
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-3)
        
        self.optimizer_zero_grad = lambda: [opt.zero_grad() for opt in [self.actor_optimizer, 
                                                                        self.critic_optimizer, 
                                                                        self.log_alpha_optimizer]]
        
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def polyak_update_target(self):
        p = self.config['agent.polyak']
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(p*target_param.data + (1 - p)*param.data)

    def choose_action(self, obs, **kwargs):
        mode = kwargs['mode']
        assert mode in ['train', 'stochastic', 'deterministic']
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        out = {}
        if mode == 'train':
            action_dist = self.actor(obs)
            action = action_dist.rsample()  # grad flow
            out['action'] = action
            out['action_logprob'] = action_dist.log_prob(action.detach())
        elif mode == 'stochastic':
            with torch.no_grad():
                action_dist = self.actor(obs)
                out['action'] = action_dist.sample().detach().cpu().numpy()
        elif mode == 'deterministic':
            with torch.no_grad():
                out['action'] = self.actor.mean_forward(obs).detach().cpu().numpy()
        else:
            raise NotImplementedError
        return out

    def learn(self, D, **kwargs):
        replay = kwargs['replay']
        episode_length = kwargs['episode_length']
        out = {}
        out['actor_loss'] = []
        out['critic_loss'] = []
        out['alpha_loss'] = []
        Q1_vals = []
        Q2_vals = []
        logprob_vals = []
        for i in range(episode_length):
            observations, actions, rewards, next_observations, masks = replay.sample(self.config['replay.batch_size'])
            
            out_actor = self.choose_action(observations, mode='train')
            sampled_actions = out_actor['action'].squeeze()
            sampled_actions_logprob = out_actor['action_logprob'].squeeze()
            
            Qs1, Qs2 = self.critic(observations, actions)
            Qs1, Qs2 = map(lambda x: x.squeeze(), [Qs1, Qs2])
            with torch.no_grad():
                next_Qs1, next_Qs2 = self.critic_target(next_observations, sampled_actions)
                next_Qs = torch.min(next_Qs1, next_Qs2).squeeze() - self.alpha*sampled_actions_logprob
                Q_targets = rewards + self.config['agent.gamma']*masks*next_Qs.detach()
            
            critic_loss = F.mse_loss(Qs1, Q_targets) + F.mse_loss(Qs2, Q_targets)
            self.optimizer_zero_grad()
            critic_loss.backward()
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['agent.max_grad_norm'])
            self.critic_optimizer.step()

            if i % self.config['agent.policy_delay'] == 0:
                actor_loss = -(self.critic.Q1(observations, sampled_actions).squeeze() - self.alpha*sampled_actions_logprob).mean()
                self.optimizer_zero_grad()
                actor_loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['agent.max_grad_norm'])
                self.actor_optimizer.step()
                
                alpha_loss = torch.mean(self.alpha*(-sampled_actions_logprob - self.target_entropy).detach())
                self.optimizer_zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()

                self.polyak_update_target()

                out['actor_loss'].append(actor_loss)
                out['alpha_loss'].append(alpha_loss)
            out['critic_loss'].append(critic_loss)
            Q1_vals.append(Qs1)
            Q2_vals.append(Qs2)
            logprob_vals.append(sampled_actions_logprob)
        out['actor_loss'] = torch.tensor(out['actor_loss']).mean().item()
        out['actor_grad_norm'] = actor_grad_norm
        out['critic_loss'] = torch.tensor(out['critic_loss']).mean().item()
        out['critic_grad_norm'] = critic_grad_norm
        describe_it = lambda x: describe(torch.cat(x).detach().cpu().numpy().squeeze(), axis=-1, repr_indent=1, repr_prefix='\n')
        out['Q1'] = describe_it(Q1_vals)
        out['Q2'] = describe_it(Q2_vals)
        out['logprob'] = describe_it(logprob_vals)
        out['alpha_loss'] = torch.tensor(out['alpha_loss']).mean().item()
        out['alpha'] = self.alpha.item()
        return out
    
    def checkpoint(self, logdir, num_iter):
        self.save(logdir/f'agent_{num_iter}.pth')
        # TODO: save normalization moments
