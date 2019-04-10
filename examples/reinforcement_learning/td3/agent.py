import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lagom import BaseAgent
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
        #ortho_init(self.action_head, weight_scale=0.01, constant_bias=0.0)
        ortho_init(self.action_head, nonlinearity='tanh', constant_bias=0.0)
        
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

        # Q1
        self.l1 = nn.Linear(flatdim(env.observation_space) + flatdim(env.action_space), 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        
        # Q2
        self.l4 = nn.Linear(flatdim(env.observation_space) + flatdim(env.action_space), 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)
        
        self.to(self.device)
        
    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2
    
    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1
    

"""
class Critic(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        
        # Q1 architecture
        self.feature_layers1 = make_fc(flatdim(env.observation_space) + flatdim(env.action_space), [400, 300])
        for layer in self.feature_layers1:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in [400, 300]])
        self.Q_head1 = nn.Linear(300, 1)
        ortho_init(self.Q_head1, weight_scale=1.0, constant_bias=0.0)
        
        # Q2 architecture
        self.feature_layers2 = make_fc(flatdim(env.observation_space) + flatdim(env.action_space), [400, 300])
        for layer in self.feature_layers2:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in [400, 300]])
        self.Q_head2 = nn.Linear(300, 1)
        ortho_init(self.Q_head2, weight_scale=1.0, constant_bias=0.0)
        
        self.to(self.device)
        
    def forward(self, x, action):
        return self.Q1(x, action), self.Q2(x, action)

    def Q1(self, x, action):
        x = torch.cat([x, action], 1)
        for layer, layer_norm in zip(self.feature_layers1, self.layer_norms1):
            x = layer_norm(F.relu(layer(x)))
        x = self.Q_head1(x)
        return x
        
    def Q2(self, x, action):
        x = torch.cat([x, action], 1)
        for layer, layer_norm in zip(self.feature_layers2, self.layer_norms2):
            x = layer_norm(F.relu(layer(x)))
        x = self.Q_head2(x)
        return x
"""
    
class Agent(BaseAgent):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(config, env, device, **kwargs)
        
        self.actor = Actor(config, env, device, **kwargs)
        self.actor_target = Actor(config, env, device, **kwargs)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        
        self.critic = Critic(config, env, device, **kwargs)
        self.critic_target = Critic(config, env, device, **kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.critic_optimizer = optim.Adam(self.critic.parameters())
        
        self.max_action = env.action_space.high[0]
        
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
            eps = np.random.normal(0.0, self.config['agent.action_noise'], size=action.shape)
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
        Q1_vals = []
        Q2_vals = []
        for i in range(episode_length):
            observations, actions, rewards, next_observations, masks = replay.sample(self.config['replay.batch_size'])
            
            next_actions = self.actor_target(next_observations)
            #eps = np.random.normal(0.0, self.config['agent.target_noise'], size=next_actions.shape)
            eps = np.random.normal(0.0, self.config['agent.target_noise'], size=1)
            eps = np.clip(eps, -self.config['agent.target_noise_clip'], self.config['agent.target_noise_clip'])
            eps = torch.from_numpy(eps).float().to(self.device)
            next_actions = torch.clamp(next_actions + eps, -self.max_action, self.max_action)
            
            Qs1, Qs2 = self.critic(observations, actions)
            Qs1, Qs2 = map(lambda x: x.squeeze(), [Qs1, Qs2])
            with torch.no_grad():
                next_Qs1, next_Qs2 = self.critic_target(next_observations, next_actions)
                next_Qs = torch.min(next_Qs1, next_Qs2).squeeze()
            targets = rewards + self.config['agent.gamma']*masks*next_Qs.detach()
            
            critic_loss = F.mse_loss(Qs1, targets) + F.mse_loss(Qs2, targets)
            #self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['agent.max_grad_norm'])
            self.critic_optimizer.step()
            
            out['critic_loss'].append(critic_loss.item())
            Q1_vals.extend(Qs1.detach().cpu().numpy())
            Q2_vals.extend(Qs2.detach().cpu().numpy())
            
            if i % self.config['agent.policy_delay'] == 0:
                actor_loss = -self.critic.Q1(observations, self.actor(observations)).mean()
                self.actor_optimizer.zero_grad()
                #self.critic_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['agent.max_grad_norm'])
                self.actor_optimizer.step()
                
                self.polyak_update_target()
            
                out['actor_loss'].append(actor_loss.item())
            
        out['actor_loss'] = np.mean(out['actor_loss'])
        out['actor_grad_norm'] = actor_grad_norm
        out['critic_loss'] = np.mean(out['critic_loss'])
        out['critic_grad_norm'] = critic_grad_norm
        out['mean_Q1'] = np.mean(Q1_vals)
        out['std_Q1'] = np.std(Q1_vals)
        out['min_Q1'] = np.min(Q1_vals)
        out['max_Q1'] = np.max(Q1_vals)
        out['mean_Q2'] = np.mean(Q2_vals)
        out['std_Q2'] = np.std(Q2_vals)
        out['min_Q2'] = np.min(Q2_vals)
        out['max_Q2'] = np.max(Q2_vals)
        return out
