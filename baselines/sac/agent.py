import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Independent
from torch.distributions import Normal
from torch.distributions import TransformedDistribution

import gym.spaces as spaces
import lagom
import lagom.utils as utils

# TODO: import from PyTorch when PR merged: https://github.com/pytorch/pytorch/pull/19785
from tanh_transform import TanhTransform


class Actor(lagom.nn.Module):
    LOGSTD_MAX = 2
    LOGSTD_MIN = -20

    def __init__(self, config, env, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        
        self.feature_layers = lagom.nn.make_fc(spaces.flatdim(env.observation_space), [256, 256])
        self.mean_head = nn.Linear(256, spaces.flatdim(env.action_space))
        self.logstd_head = nn.Linear(256, spaces.flatdim(env.action_space))

    def forward(self, x):
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        mean = self.mean_head(x)
        logstd = self.logstd_head(x)
        logstd = torch.tanh(logstd)
        logstd = self.LOGSTD_MIN + 0.5*(self.LOGSTD_MAX - self.LOGSTD_MIN)*(1 + logstd)
        std = torch.exp(logstd)
        dist = TransformedDistribution(Independent(Normal(mean, std), 1), [TanhTransform(cache_size=1)])
        return dist
    
    def mean_forward(self, x):
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        mean = self.mean_head(x)
        mean = torch.tanh(mean)
        return mean


class Critic(lagom.nn.Module):
    def __init__(self, config, env, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        
        obs_action_dim = spaces.flatdim(env.observation_space) + spaces.flatdim(env.action_space)
        # Q1
        self.Q1_layers = lagom.nn.make_fc(obs_action_dim, [256, 256])
        self.Q1_head = nn.Linear(256, 1)
        # Q2
        self.Q2_layers = lagom.nn.make_fc(obs_action_dim, [256, 256])
        self.Q2_head = nn.Linear(256, 1)

    def Q1(self, x, action):
        x = torch.cat([x, action], dim=-1)
        for layer in self.Q1_layers:
            x = F.relu(layer(x))
        x = self.Q1_head(x)
        return x
    
    def Q2(self, x, action):
        x = torch.cat([x, action], dim=-1)
        for layer in self.Q2_layers:
            x = F.relu(layer(x))
        x = self.Q2_head(x)
        return x
        
    def forward(self, x, action):
        return self.Q1(x, action), self.Q2(x, action)


class Agent(lagom.BaseAgent):
    def __init__(self, config, env, **kwargs):
        super().__init__(config, env, **kwargs)
        
        self.actor = Actor(config, env, **kwargs)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['agent.actor.lr'])
        
        self.critic = Critic(config, env, **kwargs)
        self.critic_target = Critic(config, env, **kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['agent.critic.lr'])
        
        self.target_entropy = -float(spaces.flatdim(env.action_space))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(config['agent.initial_temperature'])))
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=config['agent.actor.lr'])
        
        self.zero_grad_all = lambda: [opt.zero_grad() for opt in [self.actor_optimizer, 
                                                                  self.critic_optimizer, 
                                                                  self.log_alpha_optimizer]]
        self.register_buffer('total_timestep', torch.tensor(0))
        
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def polyak_update_target(self):
        p = self.config['agent.polyak']
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(p*target_param.data + (1 - p)*param.data)

    def choose_action(self, x, **kwargs):
        obs = torch.as_tensor(x.observation).float().to(self.config.device).unsqueeze(0)
        with torch.no_grad():
            if kwargs['mode'] == 'train':
                action = utils.numpify(self.actor(obs).sample(), 'float')
            elif kwargs['mode'] == 'eval':
                action = utils.numpify(self.actor.mean_forward(obs), 'float')
        out = {}
        out['raw_action'] = action.squeeze(0)
        return out

    def learn(self, D, **kwargs):
        replay = kwargs['replay']
        T = kwargs['T']
        list_actor_loss, list_critic_loss, list_alpha_loss, Q1_vals, Q2_vals, logprob_vals = [], [], [], [], [], []
        for i in range(T):
            samples = replay.sample(self.config['replay.batch_size'])
            observations, actions, next_observations, rewards, masks = map(lambda x: torch.as_tensor(x).to(self.config.device), samples)
            Qs1, Qs2 = self.critic(observations, actions)
            with torch.no_grad():
                action_dist = self.actor(next_observations)
                next_actions = action_dist.rsample()
                next_actions_logprob = action_dist.log_prob(next_actions).unsqueeze(-1)
                next_Qs1, next_Qs2 = self.critic_target(next_observations, next_actions)
                next_Qs = torch.min(next_Qs1, next_Qs2) - self.alpha.detach()*next_actions_logprob
                targets = rewards + self.config['agent.gamma']*masks*next_Qs
            critic_loss = F.mse_loss(Qs1, targets.detach()) + F.mse_loss(Qs2, targets.detach())
            self.zero_grad_all()
            critic_loss.backward()
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['agent.max_grad_norm'])
            self.critic_optimizer.step()
            
            action_dist = self.actor(observations)
            policy_actions = action_dist.rsample()
            policy_actions_logprob = action_dist.log_prob(policy_actions).unsqueeze(-1)
            actor_Qs1, actor_Qs2 = self.critic(observations, policy_actions)
            actor_Qs = torch.min(actor_Qs1, actor_Qs2)
            actor_loss = torch.mean(self.alpha.detach()*policy_actions_logprob - actor_Qs)
            self.zero_grad_all()
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['agent.max_grad_norm'])
            self.actor_optimizer.step()
            
            alpha_loss = torch.mean(self.log_alpha*(-policy_actions_logprob - self.target_entropy).detach())
            self.zero_grad_all()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            self.polyak_update_target()
            list_actor_loss.append(actor_loss)
            list_critic_loss.append(critic_loss)
            list_alpha_loss.append(alpha_loss)
            Q1_vals.append(Qs1)
            Q2_vals.append(Qs2)
            logprob_vals.append(policy_actions_logprob)
        self.total_timestep += T
        
        out = {}
        out['actor_loss'] = torch.as_tensor(list_actor_loss).mean(0).item()
        out['actor_grad_norm'] = actor_grad_norm
        out['critic_loss'] = torch.as_tensor(list_critic_loss).mean(0).item()
        out['critic_grad_norm'] = critic_grad_norm
        describe_it = lambda x: utils.describe(torch.cat(x).squeeze(), axis=-1, repr_indent=1, repr_prefix='\n')
        out['Q1'] = describe_it(Q1_vals)
        out['Q2'] = describe_it(Q2_vals)
        out['logprob'] = describe_it(logprob_vals)
        out['alpha_loss'] = torch.as_tensor(list_alpha_loss).mean(0).item()
        out['alpha'] = self.alpha.item()
        return out
    
    def checkpoint(self, logdir, num_iter):
        self.save(logdir/f'agent_{num_iter}.pth')
