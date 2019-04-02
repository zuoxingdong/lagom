from collections import deque

import random
import numpy as np
import torch


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
        
    def __len__(self):
        return len(self.buffer)

    def add(self, observation, action, reward, next_observation, done):
        transition = (observation, action, reward, next_observation, done)
        self.buffer.append(transition)
        
    def sample(self, batch_size):
        D = random.sample(self.buffer, batch_size)  # unique samples, rather than random.choices
        D = zip(*D)
        observations, actions, rewards, next_observations, dones = list(map(lambda x: np.asarray(x), D))
        masks = 1. - dones
        D = (observations, actions, rewards, next_observations, masks)
        D = list(map(lambda x: torch.from_numpy(x).float().to(self.device), D))
        return D
