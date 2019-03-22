import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lagom import BaseAgent
from lagom.networks import Module
from lagom.transform import LinearSchedule


class NatureDQN(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.Q_head = nn.Linear(512, env.action_space.n)
        # TODO: initialization, batchnorm?
        
        self.to(self.device)
        
    def forward(self, x):
        x = x.permute([0, 3, 1, 2])  # [batch, H, W, C] -> [batch, C, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.Q_head(x)
        return x
    
    
class RainbowDQN(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.Q_head = nn.Linear(512, env.action_space.n*config['num_atom'])
        
        self.to(self.device)
        
    def forward(self, x, support):
        x = x.permute([0, 3, 1, 2])  # [batch, H, W, C] -> [batch, C, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.Q_head(x)
        logits = x.reshape(-1, self.env.action_space.n, self.config['num_atom'])
        probs = F.softmax(logits, -1)
        q_values = (support*probs).sum(2)
        return x


class Agent(BaseAgent):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(config, env, device, **kwargs)

        self.q_net = NatureDQN(config, env, device, **kwargs)
        self.target_net = NatureDQN(config, env, device, **kwargs)
        self.sync_target_network()
        self.target_net.eval()

        self.eps_scheduler = LinearSchedule(1.0, 
                                            config['agent.eps_train'], 
                                            config['agent.eps_decay_period'], 
                                            start=config['agent.min_replay_history'])
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config['agent.lr'], eps=0.0003125)
        self.train_step = config['agent.min_replay_history']
        
    def sync_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        
    def choose_action(self, obs, **kwargs):
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        out = {}

        if self.q_net.training:
            eps = self.eps_scheduler(self.train_step)
        else:
            eps = self.config['agent.eps_eval']
        if random.random() <= eps:
            action = [self.env.action_space.sample() for _ in range(len(self.env))]
        else:
            with torch.no_grad():
                action = self.q_net(obs).argmax(1, keepdim=False).detach().cpu().numpy()

        out['action'] = action
        return out
        
    def learn(self, D=None, **kwargs):
        out = None
        
        if self.train_step % self.config['agent.update_period'] == 0:
            replay = kwargs['replay']
            observations, actions, rewards, next_observations, masks = replay.sample(self.config['replay.batch_size'])

            Qs = self.q_net(observations)[torch.arange(self.config['replay.batch_size']), actions]
            with torch.no_grad():
                next_Qs = self.target_net(next_observations).max(1)[0]
            targets = rewards + self.config['agent.gamma']*masks*next_Qs.detach()

            loss = F.smooth_l1_loss(Qs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            if self.config['agent.max_grad_norm'] is not None:
                nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config['agent.max_grad_norm'])
            self.optimizer.step()

            out = {}
            out['loss'] = loss.item()
            if self.train_step % (self.config['agent.update_period']*self.config['log.interval']) == 0:
                print(f'Train step {self.train_step}\tQ loss: {loss.item()}')
        
        if self.train_step % self.config['agent.target_update_period'] == 0:
            self.sync_target_network()
    
        self.train_step += 1
        return out
