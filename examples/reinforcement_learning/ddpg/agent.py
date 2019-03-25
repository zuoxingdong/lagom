import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lagom import BaseAgent
from lagom.envs import flatdim
from lagom.networks import Module


class Actor(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        self.fc1 = nn.Linear(flatdim(env.observation_space), 400)
        self.fc2 = nn.Linear(400, 300)
        self.action_head = nn.Linear(300, flatdim(env.action_space))
        
        assert np.unique(env.action_space.high).size == 1
        assert -np.unique(env.action_space.low).item() == np.unique(env.action_space.high).item()
        self.max_action = env.action_space.high[0]
        
        #for layer in self.feature_layers:
        #    nn.init.normal_(layer.weight, mean=0.0, std=np.sqrt(1/layer.out_features))
        #self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in config['nn.sizes']])
        
        self.to(self.device)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.max_action*torch.tanh(self.action_head(x))
        return x
        
        
class Critic(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        self.fc1 = nn.Linear(flatdim(env.observation_space), 400)
        self.fc2 = nn.Linear(400 + flatdim(env.action_space), 300)
        self.Q_head = nn.Linear(300, 1)
        
        self.to(self.device)
        
    def forward(self, x, action):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(torch.cat([x, action], dim=1)))
        x = self.Q_head(x)
        return x
    
    
class Agent(BaseAgent):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(config, env, device, **kwargs)
        
        self.actor = Actor(config, env, device, **kwargs)
        self.actor_target = Actor(config, env, device, **kwargs)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['agent.actor.lr'])
        
        self.critic = Critic(config, env, device, **kwargs)
        self.critic_target = Critic(config, env, device, **kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['agent.critic.lr'], weight_decay=1e-2)
        
        #self.total_timestep = config['']
        
        self.eps_std = 0.1
        
    def polyak_update_target(self):
        p = self.config['agent.polyak']
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(p*target_param.data + (1 - p)*param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(p*target_param.data + (1 - p)*param.data)
        
    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()
        return self
        
    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()
        return self
        
    def choose_action(self, obs, **kwargs):
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        with torch.no_grad():
            action = self.actor(obs).detach().cpu().numpy()
        if self.training:
            eps = np.random.normal(0.0, self.eps_std, size=action.shape)
            action = np.clip(action + eps, self.env.action_space.low, self.env.action_space.high)
        out = {}
        out['action'] = action
        return out
        
    def learn(self, D, **kwargs):
        replay = kwargs['replay']
        episode_length = kwargs['episode_length']
        out = {}
        out['actor_loss'] = []
        out['critic_loss'] = []
        for i in range(episode_length):
            observations, actions, rewards, next_observations, masks = replay.sample(self.config['replay.batch_size'])
            
            Qs = self.critic(observations, actions).squeeze()
            with torch.no_grad():
                next_Qs = self.critic_target(next_observations, self.actor_target(next_observations)).squeeze()
            targets = rewards + self.config['agent.gamma']*masks*next_Qs.detach()
            critic_loss = F.mse_loss(Qs, targets)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            actor_loss = -self.critic(observations, self.actor(observations)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.polyak_update_target()
            
            out['actor_loss'].append(actor_loss.item())
            out['critic_loss'].append(critic_loss.item())
        out['actor_loss'] = np.mean(out['actor_loss'])
        out['critic_loss'] = np.mean(out['critic_loss'])
        return out
