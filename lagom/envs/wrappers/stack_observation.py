import numpy as np

from lagom.envs.spaces import Box
from lagom.envs.wrappers import ObservationWrapper


class StackObservation(ObservationWrapper):
    """
    Returns the stacked observations
    """
    def __init__(self, env, num_stack):
        super().__init__(env)
        
        self.num_stack = num_stack
        
        # TODO: support Dict space
        assert isinstance(self.observation_space, Box)  # enforce as Box space
        
        # Create a new observation space
        low = np.expand_dims(self.observation_space.low, -1)
        high = np.expand_dims(self.observation_space.high, -1)
        low = np.repeat(low, self.num_stack, axis=-1)
        high = np.repeat(high, self.num_stack, axis=-1)
        dtype = self.observation_space.dtype
        self._observation_space = Box(low=low, high=high, dtype=dtype)
        
        # Initialize stacked observations
        self.stacked_observation = np.zeros(self._observation_space.shape, dtype=dtype)
        
    def reset(self):
        # Clean up all stacked observations
        self.stacked_observation.fill(0)
        
        return super().reset()
        
    def process_observation(self, observation):
        # Shift the last element to the first element for the last dimension
        self.stacked_observation = np.roll(self.stacked_observation, shift=1, axis=-1)
        # Update the first element as new observation
        self.stacked_observation[..., 0] = observation
        
        return self.stacked_observation
        
    @property
    def observation_space(self):
        return self._observation_space
