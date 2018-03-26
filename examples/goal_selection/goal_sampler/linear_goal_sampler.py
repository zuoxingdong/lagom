import numpy as np

from .base import BaseGoalSampler

        
class LinearGoalSampler(BaseGoalSampler):
    def __init__(self, goal_seq):
        self.goal_seq = goal_seq
        
        super().__init__(None, None)
        
    def sample(self):
        """Pop out the goal in order as it inserted"""
        return self.goal_seq.pop(0)