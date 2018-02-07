import numpy as np

from lagom.envs.spaces import Box
from lagom.envs.wrappers import ObservationWrapper


class GoalMaze(ObservationWrapper):
    def process_observation(self, observation):
        return np.array([*self.env.env.state, *self.env.env.goal_states[0]])
    
    @property
    def observation_space(self):
        # Update observation space
        return Box(0, self.env.get_source_env().maze_size[0]-1, shape=(4,))