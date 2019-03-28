import numpy as np

from gym.spaces import Box
from gym import ObservationWrapper


class ScaledFloatFrame(ObservationWrapper):
    r"""Convert image frame to float range [0, 1] by dividing 255. 
    
    .. warning::
    
        Do NOT use this wrapper for DQN ! It will break the memory optimization.
    
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=1, shape=self.observation_space.shape, dtype=np.float32)
    
    def observation(self, observation):
        return observation.astype(np.float32)/255.
