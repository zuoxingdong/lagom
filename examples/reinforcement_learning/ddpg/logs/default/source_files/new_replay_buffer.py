from collections import deque

import random
import numpy as np
import torch

from lagom.transform import RunningMeanVar


class ReplayBuffer(object):
    r"""A deque-based buffer of bounded size that implements experience replay for DDPG.
    
    .. note:
        Difference with DQN replay buffer: we handle raw observation (no pixel) for continuous control
        Thus we do not have transformation to and from 255. and np.uint8
    
    Args:
        capacity (int): max capacity of transition storage in the buffer. When the buffer overflows the
            old transitions are dropped.
        device (Device): PyTorch device
        
    """
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
        self.obs_moments = RunningMeanVar(shape=(17,))
        self.returns = np.zeros(1, dtype=np.float64)
        self.return_moments = RunningMeanVar(shape=())
        
    def normalize_obs(self, obs):
        if torch.is_tensor(obs):
            mean = torch.from_numpy(self.obs_moments.mean).float().to(self.device)
            var = torch.from_numpy(self.obs_moments.var).float().to(self.device)
            obs = (obs.float() - mean)/torch.sqrt(var + 1e-8)
            obs = torch.clamp(obs, -5., 5.)
        else:
            obs = (obs - self.obs_moments.mean)/np.sqrt(self.obs_moments.var + 1e-8)
            obs = np.clip(obs, -5., 5.)
        return obs
    
    def normalize_reward(self, reward):
        reward = reward/np.sqrt(self.return_moments.var + 1e-8)
        reward = np.clip(reward, -10., 10.)
        return reward
        
    def __len__(self):
        return len(self.buffer)

    def add(self, observation, action, reward, next_observation, done):
        
        self.obs_moments([observation])
        self.returns = reward + 0.99*self.returns
        self.return_moments(self.returns)
        if done:
            self.returns.fill(0.0)
        
        transition = (observation, action, reward, next_observation, done)
        self.buffer.append(transition)
        
    def sample(self, batch_size):
        D = random.sample(self.buffer, batch_size)  # unique samples, rather than random.choices
        D = zip(*D)
        observations, actions, rewards, next_observations, dones = list(map(lambda x: np.asarray(x), D))
        masks = 1. - dones
        
        observations = self.normalize_obs(observations)
        next_observations = self.normalize_obs(next_observations)
        rewards = self.normalize_reward(rewards)
        
        D = (observations, actions, rewards, next_observations, masks)
        D = list(map(lambda x: torch.from_numpy(x).float().to(self.device), D))
        return D
