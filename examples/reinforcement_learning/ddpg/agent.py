import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lagom import BaseAgent
from lagom.utils import pickle_dump
from lagom.envs import flatdim
from lagom.networks import Module
from lagom.networks import make_fc
from lagom.networks import ortho_init


class Actor(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        self.feature_layers = make_fc(flatdim(env.observation_space), [400, 300])
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in [400, 300]])
        
        self.action_head = nn.Linear(300, flatdim(env.action_space))
        ortho_init(self.action_head, weight_scale=0.01, constant_bias=0.0)
        # Layernorm here will make action large again, so don't use it !
        #self.action_layer_norm = nn.LayerNorm(flatdim(env.action_space))
        
        
        #ortho_init(self.action_head, weight_scale=0.01, constant_bias=0.0)
        #ortho_init(self.action_head, weight_scale=0.1, constant_bias=0.0)
        
        assert np.unique(env.action_space.high).size == 1
        assert -np.unique(env.action_space.low).item() == np.unique(env.action_space.high).item()
        self.max_action = env.action_space.high[0]
        
        self.to(self.device)
        
    def forward(self, x):
        for layer, layer_norm in zip(self.feature_layers, self.layer_norms):
            x = layer_norm(F.relu(layer(x)))
        x = self.max_action*torch.tanh(self.action_head(x))
        return x

    
class Critic(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        self.feature_layers = make_fc(flatdim(env.observation_space) + flatdim(env.action_space), [400, 300])
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in [400, 300]])
        
        self.Q_head = nn.Linear(300, 1)
        ortho_init(self.Q_head, weight_scale=1.0, constant_bias=0.0)
        #ortho_init(self.Q_head, weight_scale=0.01, constant_bias=0.0)
        
        
        
        self.to(self.device)
        
    def forward(self, x, action):
        x = torch.cat([x, action], dim=-1)
        for layer, layer_norm in zip(self.feature_layers, self.layer_norms):
            x = layer_norm(F.relu(layer(x)))
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
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['agent.critic.lr'])
        
        self.action_noise = config['agent.action_noise']
        
    def polyak_update_target(self):
        p = self.config['agent.polyak']
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(p*target_param.data + (1 - p)*param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(p*target_param.data + (1 - p)*param.data)

    def choose_action(self, obs, **kwargs):
        mode = kwargs['mode']
        assert mode in ['train', 'eval']
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        with torch.no_grad():
            action = self.actor(obs).detach().cpu().numpy()
        if mode == 'train':
            eps = np.random.normal(0.0, self.action_noise, size=action.shape)
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
        Q_vals = []
        for i in range(episode_length):
            observations, actions, rewards, next_observations, masks = replay.sample(self.config['replay.batch_size'])
            
            Qs = self.critic(observations, actions).squeeze()
            with torch.no_grad():
                next_Qs = self.critic_target(next_observations, self.actor_target(next_observations)).squeeze()
            targets = rewards + self.config['agent.gamma']*masks*next_Qs.detach()
            
            critic_loss = F.mse_loss(Qs, targets)
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['agent.max_grad_norm'])
            self.critic_optimizer.step()
            
            actor_loss = -self.critic(observations, self.actor(observations)).mean()
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['agent.max_grad_norm'])
            self.actor_optimizer.step()
            
            self.polyak_update_target()
            
            out['actor_loss'].append(actor_loss.item())
            out['critic_loss'].append(critic_loss.item())
            Q_vals.extend(Qs.detach().cpu().numpy())
        out['actor_loss'] = np.mean(out['actor_loss'])
        out['actor_grad_norm'] = actor_grad_norm
        out['critic_loss'] = np.mean(out['critic_loss'])
        out['critic_grad_norm'] = critic_grad_norm
        out['mean_Q'] = np.mean(Q_vals)
        out['std_Q'] = np.std(Q_vals)
        out['min_Q'] = np.min(Q_vals)
        out['max_Q'] = np.max(Q_vals)
        return out
    
    def checkpoint(self, logdir, num_iter):
        self.save(logdir/f'agent_{num_iter}.pth')
        # TODO: save normalization moments
