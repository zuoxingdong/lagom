import numpy as np
from gym.spaces import flatdim
from lagom.utils import tensorify


class ReplayBuffer(object):
    def __init__(self, env, capacity, device):
        self.env = env
        self.capacity = capacity
        self.device = device
        
        self.observations = np.zeros([capacity, flatdim(env.observation_space)], dtype=np.float32)
        self.actions = np.zeros([capacity, flatdim(env.action_space)], dtype=np.float32)
        self.rewards = np.zeros([capacity, 1], dtype=np.float32)
        self.next_observations = np.zeros([capacity, flatdim(env.observation_space)], dtype=np.float32)
        self.masks = np.zeros([capacity, 1], dtype=np.float32)
        
        self.size = 0
        self.pointer = 0
        
    def __len__(self):
        return self.size

    def _add(self, observation, action, reward, next_observation, terminal):
        self.observations[self.pointer] = observation
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_observations[self.pointer] = next_observation
        self.masks[self.pointer] = 1. - terminal
        
        self.pointer = (self.pointer+1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def add(self, traj):
        for t in range(1, traj.T+1):
            self._add(traj[t-1].observation, traj.actions[t-1], traj[t].reward, traj[t].observation, traj[t].terminal())

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return list(map(lambda x: tensorify(x, self.device), [self.observations[idx], 
                                                              self.actions[idx], 
                                                              self.rewards[idx], 
                                                              self.next_observations[idx], 
                                                              self.masks[idx]]))
