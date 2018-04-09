import numpy as np

from .base import BaseGoalSampler
from .uniform_goal_sampler import UniformGoalSampler
        
        
class RejectionGoalSampler(BaseGoalSampler):
    def __init__(self, runner, config):
        super().__init__(runner, config)
        
        self.uniform_sampler = UniformGoalSampler(self.runner, self.config)
        
    def sample(self, low=0.2, high=0.8, max_samples=10000):
        """
        Rejection sampling of goal with lower and upper bounds of goal quality measure
        
        Args:
            low (float): lower bound of goal quality measure
            high (float): upper bound of goal quality measure
            max_samples (int): maximum number of samples, avoid infinite loop
            
        Returns:
            goal (object): sampled goal between lower and upper bounds of goal quality measure
        """
        measure = -1
        num_samples = 0
        while measure < low or measure > high:
            # Uniformly sample a goal
            goal = self.uniform_sampler.sample()
            # Evaluate the quality measure of the sampled goal
            measure = self._eval_goal(goal)
            
            num_samples += 1
            if num_samples >= max_samples:
                raise ValueError('Too many samples here, please check the lower and upper bounds, avoid infinite loop.')
        
        return goal
    
    def _eval_goal(self, goal):
        """
        Evaluate the sampled goal with current agent in the environment
        
        Args:
            goal (object): given goal
        
        Returns:
            measure (float): quality measure of the goal
        """
        # Set environment with given goal 
        self.runner.env.get_source_env().goal_states = [goal]

        # Set max time steps as optimal trajectories (consistent with A* solution)
        if self.config['use_optimal_T']:
            self.config['T'] = self.runner.env.all_steps[tuple(goal)]

        # Evaluate
        # Collect one batch of data from runner
        batch_data = self.runner.run(self.config['T'], self.config['eval_num_epi'])

        # Useful metrics
        batch_returns = [np.sum(data['rewards']) for data in batch_data]
        measure = np.mean(batch_returns)  # success rate
        
        return measure