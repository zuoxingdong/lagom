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

# Parse hyperparameters from command line
parser = argparse.ArgumentParser(description='REINFORCE')
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
env.goal_states = [[4, 4]]  #[[6, 6]]

# Set random seed
env.seed(args.seed)
torch.manual_seed(args.seed)    

# Some hyperparameters
hyps = {}
num_iter = 5
T = 50  # Max time step per episode
num_episodes = 1000  # Number of episodes per training iteration
fc_sizes = [16]

# Create logger
log_dir = 'logs/'
logger = Logger(log_dir=log_dir)
        
def main():
    # Create environment specification
    env_spec = EnvSpec(env)
    # Create a goal-conditional policy
    policy = CategoricalMLPGoalPolicy(env_spec, fc_sizes=fc_sizes, predict_value=False)
    # Create an optimzer
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    # Create an agent
    agent = REINFORCEAgent(policy, optimizer)
    # Create a Runner
    runner = Runner(agent, env, args.gamma)
    
    # Data collection by interacting with environment
    log_epi_rewards = []
    for epi in range(num_iter):  # training iterations
        # Collect one data batch
        data_batch = runner.run(T, num_episodes)
        # Train agent over batch of data
        losses = agent.train(data_batch, standardize_r=True)
        
        batch_return = [np.sum(data['rewards']) for data in data_batch]
        average_return = np.mean(batch_return)
        std_return = np.std(batch_return)
        min_return = np.min(batch_return)
        max_return = np.max(batch_return)
        
        log_epi_rewards.append(average_return)
        
        # Loggings
        if epi == 0 or (epi + 1) % args.log_interval == 0:
            logger.log_train_iter(epi + 1, data_batch, losses, num_episodes)
            logger.dump_train_iter()
            
            
    
    # Save the loggings
    file_str = os.path.join(log_dir, 'seed_{:d}_lr_{:f}'.format(args.seed, args.lr))
    np.save(file_str, log_epi_rewards)
    
if __name__ == '__main__':
    main()