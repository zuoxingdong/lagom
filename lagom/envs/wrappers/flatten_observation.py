from .wrapper import ObservationWrapper


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that always returns flattened observation. """
    def process_observation(self, observation):
        return self.env.observation_space.flatten(observation)
    
    @property
    def observation_space(self):
        # Return original observation space, other functions e.g. unflatten maybe used
        return self.env.observation_space
