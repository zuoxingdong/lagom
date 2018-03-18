import numpy as np
        
        
class LinearGoalSampler(object):
    def __init__(self, goal_seq):
        self.goal_seq = goal_seq
        
    def sample(self):
        """Pop out the goal in order as it inserted"""
        return self.goal_seq.pop(0)