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
from lagom.runner import Runner

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
parser.add_argument('--log-interval', type=int, default=10,
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# Create environment
maze = RandomBlockMazeGenerator(maze_size=4, obstacle_ratio=0.0)
env = SparseMazeEnv(maze, action_type='VonNeumann', render_trace=False)
env.init_state = [1, 1]
env.goal_states = [[4, 4]]  #[[6, 6]]

# Set random seed
env.seed(args.seed)
torch.manual_seed(args.seed)    

# Some parameters
epochs = 5
max_steps_per_epi = 50  # Max time step per episode
num_epi_per_batch = 1000  # Number of episodes per data batch
fc_sizes = [16]
        
def main():
    # Create environment specification
    env_spec = {}
    env_spec['obs_dim'] = int(np.prod(env.observation_space.shape))
    env_spec['state_dim'] = None
    env_spec['action_dim'] = env.action_space.n
    env_spec['goal_dim'] = None
    
    # Create a goal-conditional policy
    env_spec['goal_dim'] = np.array(env.goal_states).reshape(-1).shape[0]
    policy = CategoricalMLPGoalPolicy(env_spec, fc_sizes=fc_sizes, predict_value=False)
    # Create an optimzer
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    # Create an agent
    agent = REINFORCEAgent(policy, optimizer)
    # Create a Runner
    runner = Runner(agent, env, args.gamma)
    
    # Data collection by interacting with environment
    log_epi_rewards = []
    for epi in range(epochs):  # training epochs
        # Collect one data batch
        data_batch = runner.run(max_steps_per_epi, num_epi_per_batch)
        # Train agent over batch of data
        losses = agent.train(data_batch, standardize_r=True)
        
        epi_rewards = np.sum(data_batch[0]['rewards'])
        
        # Loggings
        log_epi_rewards.append(epi_rewards)
        if epi == 0 or (epi + 1)%args.log_interval == 0:    
            log_str = '[Episode #{:3d}]: \n '\
                        '\t\tTotal loss: {:f}'\
                        '\t\tBatch episodic rewards: {:f}'
            print(log_str.format(epi + 1, losses['total_loss'].data[0], epi_rewards))
    
    # Save the loggings
    log_dir = 'logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_str = os.path.join(log_dir, 'seed_{:d}_lr_{:f}'.format(args.seed, args.lr))
    np.save(file_str, log_epi_rewards)
    
if __name__ == '__main__':
    main()