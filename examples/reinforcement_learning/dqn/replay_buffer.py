from collections import deque

import random
import numpy as np

import torch


class ReplayBuffer(object):
    r"""A deque-based buffer of bounded size that implements experience replay for Q-learning.
    
    Args:
        capacity (int): max capacity of transition storage in the buffer. When the buffer overflows the
            old transitions are dropped.
            
    """
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
    def __len__(self):
        return len(self.buffer)
        
    def add(self, observation, action, reward, next_observation, done):
        # Unnormalize & convert obs to uint8 to save a lot of memory, ~8x less memory
        observation, next_observation = map(lambda x: (x*255.).astype(np.uint8), [observation, next_observation])
        transition = (observation, action, reward, next_observation, done)
        self.buffer.append(transition)
        
    def sample(self, batch_size):
        r"""Sample a batch of experiences.
        
        Args:
            batch_size (int): the number of transitions to sample
            
        Returns
        -------
        observations : ndarray
            batch of observations
        actions : ndarray
            batch of actions
        rewards : ndarray
            batch of rewards
        next_observations : ndarray
            batch of next observations
        masks : ndarray
            batch of masks
        
        """
        D = random.sample(self.buffer, batch_size)  # unique samples, rather than random.choices
        D = zip(*D)
        observations, actions, rewards, next_observations, dones = list(map(lambda x: np.asarray(x), D))
        masks = 1. - dones
        # Convert obs back to float32 and normalize by 255
        observations, next_observations = map(lambda x: x.astype(np.float32)/255., [observations, next_observations])
        rewards, masks = map(lambda x: x.astype(np.float32), [rewards, masks])
        D = (observations, actions, rewards, next_observations, masks)
        D = list(map(lambda x: torch.from_numpy(x).to(self.device), D))
        return D
    
# TODO: PrioritizedReplayBuffer from baselines with SegmentTree classes
