import numpy as np

from lagom.envs.spaces import Box
from lagom.envs.wrappers import ObservationWrapper


class PartialFlattenDict(ObservationWrapper):
    """
    Returns flattened observation from a dictionary space with partial keys into a Box space. 
    """
    def __init__(self, env, keys):
        super().__init__(env)
        
        self.keys = keys
        
        spaces = self.env.observation_space.spaces
        assert all([isinstance(space, Box) for space in spaces.values()])  # enforce all Box spaces
        
        # Calculate dimensionality
        shape = (int(np.sum([spaces[key].flat_dim for key in self.keys])), )
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        
    def process_observation(self, observation):
        return np.concatenate([observation[key].ravel() for key in self.keys])
    
    @property
    def observation_space(self):
        return self._observation_space
