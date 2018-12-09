import numpy as np
import cv2

from lagom.envs.spaces import Box
from .wrapper import ObservationWrapper


class ResizeObservation(ObservationWrapper):
    r"""Downsample the image observation to a square image. """
    def __init__(self, env, size):
        super().__init__(env)
        assert size > 0
        self.size = size
        
    def process_observation(self, observation):
        observation = cv2.resize(observation, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation
        
    @property
    def observation_space(self):
        obs_shape = list(self.env.observation_space.shape)
        obs_shape[0] = self.size
        obs_shape[1] = self.size
        return Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
