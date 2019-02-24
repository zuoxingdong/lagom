from gym import ObservationWrapper

from lagom.envs import flatten


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation. 
    
    .. note:
    
        Keep the original observation space, because e.g. unflatten maybe used
    
    """
    def observation(self, observation):
        return flatten(self.observation_space, observation)
