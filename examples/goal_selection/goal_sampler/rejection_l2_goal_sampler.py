import numpy as np

from .base import BaseGoalSampler
from .rejection_goal_sampler import RejectionGoalSampler


class RejectionL2GoalSampler(BaseGoalSampler):
    def __init__(self, runner, config):
        super().__init__(runner, config)
        
        self.rejection_sampler = RejectionGoalSampler(self.runner, self.config)
        
    def sample(self, low=0.3, high=0.7, group_size=10, max_samples=1000):
        """
        Rejection sampling of group of goals with lower and upper bounds of goal quality measure. Then select a goal which shorest L2 distance to initial position
        
        Args:
            low (float): lower bound of goal quality measure
            high (float): upper bound of goal quality measure
            group_size (int): number of rejection sampling to be considered
            max_samples (int): maximum number of samples for rejection sampling, avoid infinite loop
            
        Returns:
            goal (object): sampled goal between lower and upper bounds of goal quality measure and the goal closest to initial position
        """
        goals = []
        for i in range(group_size):
            goal = self.rejection_sampler.sample(low, high, max_samples)
            goals.append(goal)
            
        # Compute L2 norms of all the goals compared with initial state
        l2_norms = [self._measure(goal) for goal in goals]
        # Choose the goal with smallest L2 norm
        goal = goals[np.argmin(l2_norms)]
        
        return goal
    
    def _measure(self, goal):
        init = np.array(self.runner.env.get_source_env().init_state)
        goal = np.array(goal)
        l2 = np.linalg.norm(goal - init)
        
        return l2
        