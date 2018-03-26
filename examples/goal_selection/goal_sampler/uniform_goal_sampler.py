import numpy as np

from .base import BaseGoalSampler

        
class UniformGoalSampler(BaseGoalSampler):
    def __init__(self, runner, config):
        super().__init__(runner, config)
        
        self.env = self.runner.env
        
        # Get all indicies for free locations in state space
        self.free_space = np.where(self.env.get_source_env().maze == 0)
        self.free_space = list(zip(self.free_space[0], self.free_space[1]))
        
        # Define goal space as free state space
        self.goal_space = self.free_space
        
    def sample(self):
        # Uniformly sample a goal from goal space
        idx = np.random.choice(len(self.goal_space))
        goal = list(self.goal_space[idx])
        
        return goal