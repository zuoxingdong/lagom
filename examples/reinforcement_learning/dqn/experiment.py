import os
from pathlib import Path
from itertools import count
from collections import deque

import gym

from lagom import RandomAgent

from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds

from lagom.experiment import Config
from lagom.experiment import Grid
from lagom.experiment import Sample
from lagom.experiment import run_experiment

from lagom.envs import make_vec_env
from lagom.envs import make_atari

from lagom.runner import EpisodeRunner

from agent import Agent
from engine import Engine
from replay_buffer import ReplayBuffer


config = Config(
    {'cuda': True, 
     'log.dir': 'logs/default', 
     'log.interval': 10000, 
     
     'env.id': Grid(['Pong']), # ['Breakout', 'SpaceInvaders']
     'env.max_episode_steps': 27000,
     
     'agent.gamma': 0.99,
     'agent.update_horizon': 1,
     'agent.min_replay_history': 20000,  # initialize replay buffer with random transitions
     'agent.update_period': 4,  # period betwen DQN updates
     'agent.target_update_period': 8000,  # period to update target network by copy latest Q-network
     'agent.eps_train': 0.01,  # min eps during training
     'agent.eps_eval': 0.001,  # eps fixed during evaluation
     'agent.eps_decay_period': 250000,  # length of the epsilon decay schedule
     'agent.lr': 2.5e-4,
     'agent.max_grad_norm': 10.0,  # grad clipping by norm
     
     'replay.capacity': 1000000,
     'replay.batch_size': 32,
     
     'train.iter': 100,#200,
     'train.timestep': 250000,  # number of steps per iteration
     'eval.timestep': 125000
     
    })


def initialize_replay(config, env, replay):
    print('Initializing replay buffer...')
    random_agent = RandomAgent(None, env, None)
    observation = env.reset()
    for t in range(config['agent.min_replay_history']):
        action = random_agent.choose_action(observation)['raw_action']
        next_observation, reward, done, info = env.step(action)
        if done[0]:  # single environment
            terminal_observation = info[0]['terminal_observation']
            replay.add(observation[0], action[0], reward[0], terminal_observation, done[0])
        else:
            replay.add(observation[0], action[0], reward[0], next_observation[0], done[0])
        observation = next_observation
    print('Done')


def run(config, seed, device):
    set_global_seeds(seed)
    logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)
    
    def make_env():
        return make_atari(config['env.id'], sticky_action=True, max_episode_steps=config['env.max_episode_steps'])
    env = make_vec_env(make_env, 1, seed, 'serial')  # single environment
    
    agent = Agent(config, env, device)
    replay = ReplayBuffer(config['replay.capacity'], device)
    initialize_replay(config, env, replay)
    engine = Engine(config, agent=agent, env=env, replay=replay)
    running_rewards = deque(maxlen=100)
    
    train_logs = []
    for n in range(config['train.iter']):
        train_logger = engine.train(n, running_rewards=running_rewards)
        train_logs.append(train_logger.logs)
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    return None  
    

if __name__ == '__main__':
    run_experiment(run=run, 
                   config=config, 
                   seeds=[4153361530], #, 3503522377, 2876994566, 172236777, 3949341511, 849059707], 
                   num_worker=os.cpu_count())