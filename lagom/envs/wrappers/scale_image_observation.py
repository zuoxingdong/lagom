import numpy as np

from lagom.envs.spaces import Box
from .wrapper import ObservationWrapper


class ScaleImageObservation(ObservationWrapper):
    r"""Convert float range to [0, 1] by dividing 255. """
    def process_observation(self, observation):
        return observation.astype(np.float32)/255
    
    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=self.env.observation_space.shape, dtype=np.float32)
