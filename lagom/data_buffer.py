import numpy as np
from gym.spaces import flatdim
from torch.utils import data


class UniformTransitionBuffer(data.Dataset):
    def __init__(self, env, capacity):
        self.env = env
        self.capacity = capacity

        self.observations = np.zeros([capacity, flatdim(env.observation_space)], dtype=np.float32)
        self.actions = np.zeros([capacity, flatdim(env.action_space)], dtype=np.float32)
        self.next_observations = np.zeros([capacity, flatdim(env.observation_space)], dtype=np.float32)
        self.rewards = np.zeros([capacity, 1], dtype=np.float32)
        self.masks = np.zeros([capacity, 1], dtype=np.float32)
        
        self.size = 0
        self.pointer = 0

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.observations[index], self.actions[index], self.next_observations[index], self.rewards[index], self.masks[index]

    def add_transition(self, observation, action, next_observation, reward, terminal):
        self.observations[self.pointer] = observation
        self.actions[self.pointer] = action
        self.next_observations[self.pointer] = next_observation
        self.rewards[self.pointer] = reward
        self.masks[self.pointer] = 1. - terminal

        self.pointer = (self.pointer+1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_trajectory(self, traj):
        for t in range(traj.T):
            observation = traj[t].observation
            action = traj.actions[t]
            next_observation = traj[t+1].observation
            reward = traj[t+1].reward
            terminal = traj[t+1].terminal()
            self.add_transition(observation, action, next_observation, reward, terminal)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.observations[idx], self.actions[idx], self.next_observations[idx], self.rewards[idx], self.masks[idx]

    def __repr__(self):
        return f'{self.__class__.__name__}(size: {self.size}, capacity: {self.capacity})'
