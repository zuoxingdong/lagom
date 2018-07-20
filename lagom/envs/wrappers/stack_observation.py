import numpy as np

from lagom.envs.spaces import Box

from lagom.envs.wrappers import ObservationWrapper


class StackObservation(ObservationWrapper):
    """
    The observations are stacked, e.g. if the number of stacks is 4, then the returned
    observation contains the most recent 4 observations. 
    
    Each time the step() is called, the stacked observation is updated by augmenting the
    new observation and remove the oldest one. 
    
    Note that the observation sapce is enforced to be Box to make sure the observation is an array. 
    
    For example in 'Pendulum-v0', the observation is an array with the shape [3], if we stacks 4
    observations, each step(), the returned observation will have shape [3, 4]
    """
    def __init__(self, env, num_stack):
        """
        Args:
            env (Env): environment object
            num_stack (int): number of stacks for the observation
        """
        super().__init__(env)
        
        self.num_stack = num_stack
        
        # TODO: support Dict space
        assert isinstance(env.observation_space, Box)  # enforce as Box space
        
        # Create a new observation space
        low = np.repeat(env.observation_space.low[..., np.newaxis], self.num_stack, axis=-1)
        high = np.repeat(env.observation_space.high[..., np.newaxis], self.num_stack, axis=-1)
        dtype=env.observation_space.dtype
        self.stacked_observation_space = Box(low=low, high=high, dtype=dtype)
        
        # Initialize stacked observations
        self.stacked_observation = np.zeros(self.stacked_observation_space.shape, dtype=dtype)
        
    def reset(self):
        # Clean up all stacked observation
        self.stacked_observation.fill(0)
        
        # Call reset in original environment
        return super().reset()
        
    def process_observation(self, observation):
        # Shift the oldest observation to the front
        self.stacked_observation = np.roll(self.stacked_observation, shift=1, axis=-1)
        # Change the front as new observation
        self.stacked_observation[..., 0] = observation
        
        return self.stacked_observation
    
    @property
    def observation_space(self):
        return self.stacked_observation_space
