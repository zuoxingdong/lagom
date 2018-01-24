import gym

from gym_maze.envs import SparseMazeEnv
from gym_maze.envs import RandomBlockMazeGenerator

import numpy as np

from argparse import Namespace

from multiprocessing import Process

import torch

from lagom.experiment.base import BaseExperiment
from lagom.core.utils import Logger
from lagom.core.plotter import Plotter
##############
# TODO: change name to goal selection experiment and move to goal selection folder
##############
class SimpleExperiment(BaseExperiment):
    def configure(self, num_configs):
        list_args = []
        for i in range(num_configs):
            args = Namespace()
            
            # Manual settings
            args.ID = i  # unique job ID (unique settings), e.g. unique logging file
            
            args.gamma = 0.99
            args.num_iter = 5
            args.num_episodes = 100  # Number of episodes per training iteration
            args.T = 100  # Max time step per episode
            
            args.fc_sizes = [16]
            args.predict_value = False
            args.standardize_r = True
            
            args.render = False
            
            args.log_interval = 1
            
            args.num_outer_iter = 3  # length of sequence of goals to train
            
            # Random settings
            args.seed = np.random.randint(low=0, high=99999)
            args.lr = 1e-2
            #args.lr = 10**np.random.uniform(low=-6, high=1)  # numerical stability, see http://cs231n.github.io/neural-networks-3/#hyper
            
            # Record the configuration
            list_args.append(args)
            
        return list_args
            
    def benchmark(self, num_process=1):
        if num_process > len(self.list_args):
            raise ValueError('The number of process should not be larger than the number of configurations.')   
        
        # Create environment
        maze = RandomBlockMazeGenerator(maze_size=4, obstacle_ratio=0.0)
        env = SparseMazeEnv(maze, action_type='VonNeumann', render_trace=False)
        env.init_state = [1, 1]
        #env.goal_states = [[4, 4]]
        
        # Create batches of args to run in parallel with Process
        for i in range(0, len(self.list_args), num_process):
            batch_args = self.list_args[i : i+num_process]
            
            list_process = []
            # Run experiments for the batched args, each with an individual Process
            for args in batch_args:
                print('{:#^50}'.format('#'))
                print('# Job ID: {:<10}'.format(args.ID))
                print('{:#^50}'.format('#'))

                # Set random seed
                env.seed(args.seed)
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)

                # Run algorithm for specific configuration
                process = Process(target=self.algo.run, args=[env, args])
                process.start()
                
                list_process.append(process)
                
            # Join the processes
            [process.join() for process in list_process]
