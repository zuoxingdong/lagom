import numpy as np
import gym

from lagom.transform import RunningMeanVar


class NormalizeReward(gym.RewardWrapper):
    def __init__(self, env, clip=10., gamma=0.99, constant_var=None):
        super().__init__(env)
        self.clip = clip
        assert gamma > 0.0 and gamma < 1.0, 'we do not allow discounted factor as 1.0. See docstring for details. '
        self.gamma = gamma
        self.constant_var = constant_var
        self.eps = 1e-8
        if constant_var is None:
            self.reward_moments = RunningMeanVar(shape=())
        
        # Buffer to save discounted returns from each environment
        self.all_returns = 0.0
        
    def reset(self):
        # Reset returns buffer
        self.all_returns = 0.0
        return super().reset()
    
    def step(self, action):
        observation, reward, done, info = super().step(action)
        # Set discounted return buffer as zero if episode terminates
        if done:
            self.all_returns = 0.0
        return observation, reward, done, info
    
    def reward(self, reward):
        if self.constant_var is None:
            self.all_returns = reward + self.gamma*self.all_returns
            self.reward_moments([self.all_returns])
            std = np.sqrt(self.reward_moments.var + self.eps)
        else:
            std = np.sqrt(self.constant_var + self.eps)
        # Do NOT subtract from mean, but only divided by std
        reward = np.clip(reward/std, -self.clip, self.clip)
        return reward
