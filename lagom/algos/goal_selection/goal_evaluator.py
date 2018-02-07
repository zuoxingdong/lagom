import numpy as np

from lagom.core.utils import Logger
from lagom.evaluator import BaseEvaluator
from lagom.evaluator import SimpleEvaluator

    
class GoalEvaluator(BaseEvaluator):
    def evaluate(self):
        # Get all indicies for free locations in state space
        free_space = np.where(self.runner.env.get_source_env().maze == 0)
        free_space = list(zip(free_space[0], free_space[1]))
        
        # Define goal space as free state space
        goal_space = free_space

        # Evaluate the performance of current agent over all feasible goals
        for goal in goal_space:
            # Set environment with specific goal 
            self.runner.env.get_source_env().goal_states = [goal]
            
            # Evaluate
            eval_logger = Logger(path=self.logger.path, dump_mode=[])
            evaluator = SimpleEvaluator(self.agent, self.runner, self.args, eval_logger)
            evaluator.evaluate()
            
            # Obtain and log metrics
            if self.logger is not None:
                average_return = eval_logger.metriclog['Average Return']
                self.logger.log_metric('Average Return over all goals', average_return, goal)
            
            