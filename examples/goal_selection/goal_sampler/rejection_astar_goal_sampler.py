from examples.goal_selection.utils import get_optimal_steps

from .rejection_l2_goal_sampler import RejectionL2GoalSampler


class RejectionAstarGoalSampler(RejectionL2GoalSampler):
    def _measure(self, goal):
        # Set environment with given goal 
        self.runner.env.get_source_env().goal_states = [goal]
        
        # Calculate the optimal number of steps by A* planner
        optimal_steps = get_optimal_steps(self.runner.env)
        
        return optimal_steps