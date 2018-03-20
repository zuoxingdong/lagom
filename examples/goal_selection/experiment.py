from gym_maze.envs import MazeEnv
from gym_maze.envs import UMazeGenerator

from lagom.experiment import BaseExperiment
from lagom.experiment import GridConfig
from lagom.envs import GymEnv
from lagom.envs.wrappers import SparseReward

import torch.nn.functional as F

from utils import GoalMaze


class Experiment(BaseExperiment):
    def _configure(self):
        config = GridConfig()
        
        config.add('seed', list(range(10)))  # random seeds
        
        config.add('hidden_sizes', [16])
        config.add('hidden_nonlinearity', [F.relu])
        config.add('lr', [1e-2])  # learning rate of policy network
        config.add('gamma', [0.99])  # discount factor
        config.add('T', [100])  # Max time step per episode
        config.add('use_optimal_T', [True])  # True: args.T will be modified to optimal steps before rollout for each new goal
        config.add('predict_value', [False])
        config.add('standardize_r', [True])
        
        config.add('num_outer_iter', [5])  # length of sequence of goals to train
        config.add('num_iter', [1])  # number of training iterations for each goal
        config.add('num_episodes', [1])  # Number of episodes per training iteration
        config.add('eval_num_episodes', [10])  # Number of episodes per evaluation iteration
        
        config.add('render', [False])
        config.add('log_interval', [1])
        
        return config.make_configs()
            
    def _make_env(self):
        # Create environment
        maze = UMazeGenerator(len_long_corridor=5, len_short_corridor=2, width=2, wall_size=1)
        env = MazeEnv(maze, action_type='VonNeumann', render_trace=False)
        env = GymEnv(env)  # Gym wrapper
        env = GoalMaze(env)  # flattened observation (coordiantes) augmented with goal coordinates
        env = SparseReward(env)  # sparse reward function
        
        # Set fixed initial state
        env.get_source_env().init_state = [6, 1]
        
        return env
