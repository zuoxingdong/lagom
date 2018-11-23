import numpy as np

from .wrapper import ObservationWrapper
from lagom.envs.spaces import Box


class TimeAwareObservation(ObservationWrapper):
    r"""Augment the observation with current time step in the trajectory. 
    
    .. note::
        Currently it only works with one-dimensional observation space. It doesn't
        support pixel observation space yet. 
    
    """
    def process_observation(self, observation):
        return np.append(observation, self.t)
        
    def step(self, action):
        self.t += 1
        return super().step(action)
        
    def reset(self):
        self.t = 0
        return super().reset()
        
    @property
    def observation_space(self):
        low = np.append(self.env.observation_space.low, 0.0)
        high = np.append(self.env.observation_space.high, np.inf)
        return Box(low, high, np.float32)
