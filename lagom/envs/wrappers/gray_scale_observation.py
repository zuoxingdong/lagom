import numpy as np
import cv2

from lagom.envs.spaces import Box
from .wrapper import ObservationWrapper


class GrayScaleObservation(ObservationWrapper):
    r"""Convert the image observation from RGB to gray scale. """
    def process_observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = np.expand_dims(observation, -1)
        return observation
    
    @property
    def observation_space(self):
        obs_shape = self.env.observation_space.shape[:2]
        return Box(low=0, high=255, shape=(*obs_shape, 1), dtype=np.uint8)
