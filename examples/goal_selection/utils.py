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
    
    
def get_optimal_steps(env):
    env.reset()

    # Solve maze by A* search from current state to goal
    solver = AstarSolver(env.get_source_env(), env.get_source_env().goal_states[0])

    if not solver.solvable():
        raise Error('The maze is not solvable given the current state and the goal state')

    num_optimal_steps = len(solver.get_actions())

    if num_optimal_steps == 0:  # for initial state, A* gives 0 step leading to numerical problem.
        num_optimal_steps = 2  # For initial position, optimally 2 steps needed

    return num_optimal_steps
