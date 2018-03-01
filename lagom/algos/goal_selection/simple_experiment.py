import numpy as np

from argparse import Namespace

from gym_maze.envs import MazeEnv
from gym_maze.envs import UMazeGenerator

from lagom import BaseExperiment
from lagom.envs import GymEnv
from lagom.envs.wrappers import SparseReward

from lagom.algos.goal_selection.utils import GoalMaze


class SimpleExperiment(BaseExperiment):
    def _configure(self, num_configs):
        list_args = []
        for i in range(num_configs):
            args = Namespace()
            
            # Manual settings
            args.ID = i  # unique job ID (unique settings), e.g. unique logging file
            
            args.gamma = 0.99
            
            args.num_outer_iter = 10000  # length of sequence of goals to train
            args.num_iter = 1  # number of training iterations for each goal
            args.num_episodes = 1#50  # Number of episodes per training iteration
            args.eval_num_episodes = 10  # Number of episodes per evaluation iteration
            
            args.T = 100  # Max time step per episode
            args.use_optimal_T = True#True  # True: args.T will be modified to optimal steps before rollout for each new goal
            
            args.fc_sizes = [16]
            args.predict_value = False
            args.standardize_r = True
            
            args.render = False
            
            args.log_interval = 1
            
            # Random settings
            args.seed = np.random.randint(low=0, high=99999999)
            args.lr = 1e-2
            #args.lr = 10**np.random.uniform(low=-6, high=1)  # numerical stability, see http://cs231n.github.io/neural-networks-3/#hyper
            
            # Record the configuration
            list_args.append(args)
            
        return list_args
            
    def _make_env(self, settings=None):
        # Create environment
        maze = UMazeGenerator(len_long_corridor=5, len_short_corridor=2, width=2, wall_size=1)
        env = MazeEnv(maze, action_type='VonNeumann', render_trace=False)
        env = GymEnv(env)  # Gym wrapper
        env = GoalMaze(env)  # flattened observation (coordiantes) augmented with goal coordinates
        env = SparseReward(env)  # sparse reward function
        
        # Set fixed initial state
        env.get_source_env().init_state = [6, 1]
        
        return env
