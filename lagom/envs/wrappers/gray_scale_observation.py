import numpy as np
import cv2

from lagom.envs.spaces import Box
from .wrapper import ObservationWrapper


class GrayScaleObservation(ObservationWrapper):
    r"""Convert the image observation from RGB to gray scale. """
    def __init__(self, env, keep_dim=False):
        super().__init__(env)
        
        self.keep_dim = keep_dim
        
    def process_observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        return observation
    
    @property
    def observation_space(self):
        obs_shape = self.env.observation_space.shape[:2]
        if self.keep_dim:
            return Box(low=0, high=255, shape=(*obs_shape, 1), dtype=np.uint8)
        else:
            return Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
