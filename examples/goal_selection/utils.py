import numpy as np

from gym_maze.envs.Astar_solver import AstarSolver

from lagom.envs.spaces import Box
from lagom.envs.wrappers import ObservationWrapper


class GoalMaze(ObservationWrapper):
    def process_observation(self, observation):
        return np.array([*self.env.env.state, *self.env.env.goal_states[0]])
    
    @property
    def observation_space(self):
        # Update observation space
        return Box(0, self.env.get_source_env().maze_size[0]-1, shape=(4,), dtype=np.float32)
    
    
def get_all_optimal_steps(env):
    # Create dictionary stores optimal time steps for all goals
    all_steps = {}
    
    # Get all indicies for free locations in state space
    free_space = np.where(env.get_source_env().maze == 0)
    free_space = list(zip(free_space[0], free_space[1]))
    # Define goal space as free state space
    goal_space = free_space
    
    for goal in goal_space:
        # Set the goal to the environment
        env.get_source_env().goal_states = [goal]
        
        # Reset the environment
        env.reset()

        # Solve maze by A* search from current state to goal
        solver = AstarSolver(env.get_source_env(), env.get_source_env().goal_states[0])
        if not solver.solvable():
            raise Error('The maze is not solvable given the current state and the goal state')

        num_optimal_steps = len(solver.get_actions())
        if num_optimal_steps == 0:  # for initial state, A* gives 0 step leading to numerical problem.
            num_optimal_steps = 2  # For initial position, optimally 2 steps needed
            
        all_steps[tuple(goal)] = num_optimal_steps

    return all_steps
