import gym
from gym import spaces
import numpy as np


# TODO: remove it after new wrapper merged into gym officially
class NormalizeAction(gym.ActionWrapper):
    r"""Rescale the continuous action space of the environment from [-1, 1]. """
    def __init__(self, env):
        assert isinstance(env.action_space, spaces.Box), 'expected Box action space.'
        super().__init__(env)

    def action(self, action):
        assert np.all(action >= -1.0) and np.all(action <= 1.0), 'expected range within [-1, 1], use tanh'
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (1.0 + action)*(high - low)/2.0
        action = np.clip(action, low, high)
        return action
