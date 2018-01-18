import gym

from gym_maze.envs import SparseMazeEnv
from gym_maze.envs import RandomBlockMazeGenerator

import numpy as np

import argparse

import os

import torch
import torch.optim as optim

########
# TODO: make setup.py, then this file to examples folder 
########
from lagom.agents import REINFORCEAgent
from lagom.core.policies import CategoricalMLPGoalPolicy
from lagom.core.utils import Logger
from lagom.runner import Runner
from lagom.envs import EnvSpec
from lagom.trainer import SimpleTrainer

# Parse hyperparameters from command line
parser = argparse.ArgumentParser(description='REINFORCE')
parser.add_argument('--ID', type=int, default=1, 
                    help='Unique job ID, e.g. for a unique hyperparameter setting')
parser.add_argument('--lr', type=float, default=1e-2, 
                    help='learning rate (default: 1e-2)')
parser.add_argument('--gamma', type=float, default=0.99, 
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, 
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', 
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1,
                    help='interval between training status logs (default: 1)')
args = parser.parse_args()

# Create environment
maze = RandomBlockMazeGenerator(maze_size=4, obstacle_ratio=0.0)
env = SparseMazeEnv(maze, action_type='VonNeumann', render_trace=False)
env.init_state = [1, 1]
env.goal_states = [[4, 4]]

# Set random seed
env.seed(args.seed)
torch.manual_seed(args.seed)    

# More manual settings/hyperparameters
args.num_iter = 5
args.T = 50  # Max time step per episode
args.num_episodes = 1000  # Number of episodes per training iteration
args.fc_sizes = [16]
args.predict_value = False
args.standardize_r = True

# Create logger
logger = Logger(path='logs/', dump_mode=['screen'])
        
def main():
    # Create environment specification
    env_spec = EnvSpec(env)
    # Create a goal-conditional policy
    policy = CategoricalMLPGoalPolicy(env_spec, fc_sizes=args.fc_sizes, predict_value=args.predict_value)
    # Create an optimzer
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    # Create an agent
    agent = REINFORCEAgent(policy, optimizer)
    # Create a Runner
    runner = Runner(agent, env, args.gamma)
    
    # Training phase
    trainer = SimpleTrainer(agent, runner, args, logger)
    trainer.train()
    
    # Save all loggings to a .npy file
    np.save(os.path.join(logger.get_path(), 'ID_{:d}'.format(args.ID)), 
            logger.get_metriclog())
    
if __name__ == '__main__':
    main()
    
    
    
    
    

        


