from gym_maze.envs import MazeEnv
from gym_maze.envs import UMazeGenerator

from lagom.experiment import BaseExperiment
from lagom.experiment import GridConfig
from lagom.envs import GymEnv
from lagom.envs.wrappers import SparseReward

import torch.nn.functional as F

from utils import GoalMaze
from utils import get_all_optimal_steps

from goal_sampler import UniformGoalSampler
from goal_sampler import RejectionGoalSampler
from goal_sampler import RejectionL2GoalSampler
from goal_sampler import RejectionAstarGoalSampler
from goal_sampler import SWUCBgGoalSampler


class Experiment(BaseExperiment):
    def _configure(self):
        config = GridConfig()
        
        config.add('seed', list(range(1)))  # random seeds
        
        config.add('hidden_sizes', [[16]])
        config.add('hidden_nonlinearity', [F.tanh])
        config.add('lr', [1e-2])  # learning rate of policy network
        config.add('gamma', [0.99])  # discount factor
        config.add('T', [30])  # Max time step per episode
        config.add('use_optimal_T', [False])  # True: args.T will be modified to optimal steps before rollout for each new goal
        config.add('predict_value', [False])  # Value function head
        config.add('standardize_r', [True])  # standardize returns in [-1, 1], more stable learning
        
        config.add('goal_sampler', [UniformGoalSampler])  # different goal samplers
        
        config.add('num_goal', [1000])  # length of sequence of goals to train
        config.add('train_iter', [1])  # number of training iterations
        config.add('eval_iter', [1])  # number of evaluation iterations
        config.add('train_num_epi', [5])  # Number of episodes per training iteration
        config.add('eval_num_epi', [10])  # Number of episodes per evaluation iteration
        
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
        
        # Compute all optimal steps according to A*
        env.all_steps = get_all_optimal_steps(env)
        
        return env
