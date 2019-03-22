import numpy as np

import gym
from gym.spaces import Box


class SanityEnv(gym.Env):
    def __init__(self):
        self.T = np.random.randint(4, 6+1)
        
        self.reward_range = (0.1, float('inf'))
        self.observation_space = Box(10, 60, shape=(), dtype=np.float32)
        self.action_space = Box(1, 6, shape=(1,), dtype=np.float32)
        self.spec = gym.envs.registration.EnvSpec('Sanity-v0', max_episode_steps=self.T)
        
    def step(self, action):
        self.t += 1
        self.s += 10.0
        self.r += 0.1
        done = self.t == self.T
        
        return self.s, self.r, done, {}
        
    def reset(self):
        self.s = 0.0
        self.r = 0.0
        self.t = 0
        
        return self.s
        
    def seed(self, seed):
        return [seed]
        
    def render(self, mode='human'):
        pass
        
    def close(self):
        pass
