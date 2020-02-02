from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym.spaces as spaces
import lagom
import lagom.utils as utils


class Actor(lagom.nn.Module):
    def __init__(self, config, env, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        
        self.feature_layers = lagom.nn.make_fc(spaces.flatdim(env.observation_space), config['agent.actor.sizes'])
        self.action_head = nn.Linear(config['agent.actor.sizes'][-1], spaces.flatdim(env.action_space))

    def forward(self, x):
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        x = torch.tanh(self.action_head(x))
        return x


class Critic(lagom.nn.Module):
    def __init__(self, config, env, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        
        obs_action_dim = spaces.flatdim(env.observation_space) + spaces.flatdim(env.action_space)
        self.Q_layers = lagom.nn.make_fc(obs_action_dim, config['agent.critic.sizes'])
        self.Q_head = nn.Linear(config['agent.critic.sizes'][-1], 1)
        
    def forward(self, x, action):
        x = torch.cat([x, action], dim=-1)
        for layer in self.Q_layers:
            x = F.relu(layer(x))
        x = self.Q_head(x)
        return x
    
    
class Agent(lagom.BaseAgent):
    def __init__(self, config, env, **kwargs):
        super().__init__(config, env, **kwargs)
        
        self.actor = Actor(config, env, **kwargs)
        self.actor_target = Actor(config, env, **kwargs)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['agent.actor.lr'])
        
        self.critic = Critic(config, env, **kwargs)
        self.critic_target = Critic(config, env, **kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['agent.critic.lr'])
        
        self.zero_grad_all = lambda: [opt.zero_grad() for opt in [self.actor_optimizer, self.critic_optimizer]]
        self.register_buffer('total_timestep', torch.tensor(0))

    def polyak_update_target(self):
        p = self.config['agent.polyak']
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(p*target_param.data + (1 - p)*param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(p*target_param.data + (1 - p)*param.data)

    def choose_action(self, x, **kwargs):
        obs = torch.as_tensor(x.observation).float().to(self.config.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs)
            if kwargs['mode'] == 'train':
                eps = self.config['agent.action_noise']*torch.randn_like(action)
                action = torch.clamp(action + eps, -1.0, 1.0)
        out = {}
        out['raw_action'] = utils.numpify(action.squeeze(0), 'float')
        return out

    def learn(self, D, **kwargs):
        replay, T = map(kwargs.get, ['replay', 'T'])
        out = defaultdict(list)
        for i in range(T):
            samples = replay.sample(self.config['replay.batch_size'])
            observations, actions, next_observations, rewards, masks = map(lambda x: torch.as_tensor(x).to(self.config.device), samples)
            
            Qs = self.critic(observations, actions)
            with torch.no_grad():
                next_Qs = self.critic_target(next_observations, self.actor_target(next_observations))
                targets = rewards + self.config['agent.gamma']*masks*next_Qs    
            critic_loss = F.mse_loss(Qs, targets.detach())
            self.zero_grad_all()
            critic_loss.backward()
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), torch.finfo(torch.float32).max)
            self.critic_optimizer.step()
            
            actor_loss = -self.critic(observations, self.actor(observations)).mean()
            self.zero_grad_all()
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), torch.finfo(torch.float32).max)
            self.actor_optimizer.step()
            
            self.polyak_update_target()
            out['actor_loss'].append(actor_loss)
            out['critic_loss'].append(critic_loss)
            out['Q'].append(Qs)
        self.total_timestep += T
        
        out['actor_loss'] = torch.as_tensor(out['actor_loss']).mean(0).item()
        out['actor_grad_norm'] = actor_grad_norm
        out['critic_loss'] = torch.as_tensor(out['critic_loss']).mean(0).item()
        out['critic_grad_norm'] = critic_grad_norm
        out['Q'] = utils.describe(torch.cat(out['Q']).squeeze(), axis=-1, repr_indent=1, repr_prefix='\n')
        return out
    
    def checkpoint(self, logdir, num_iter):
        self.save(logdir/f'agent_{num_iter}.pth')
