import os
from pathlib import Path
from itertools import count
from collections import deque

import gym
from gym.spaces import Box

from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds

from lagom.experiment import Config
from lagom.experiment import Grid
from lagom.experiment import Sample
from lagom.experiment import run_experiment

from lagom.envs import make_vec_env
from lagom.envs.wrappers import TimeAwareObservation
from lagom.envs.wrappers import ClipAction
from lagom.envs.wrappers import VecMonitor
from lagom.envs.wrappers import VecStandardizeObservation
from lagom.envs.wrappers import VecStandardizeReward

from lagom.runner import EpisodeRunner

from agent import Agent
from engine import Engine
from replay_buffer import ReplayBuffer


config = Config(
    {'cuda': True, 
     'log.dir': 'logs/default_', 
     'log.interval': 1, 
     
     'env.id': Grid(['HalfCheetah-v2']),  #['Hopper-v2', 'Ant-v2']
     'env.standardize': False,  # use VecStandardize
     'env.time_aware_obs': False,  # append time step to observation
     
     'agent.gamma': 0.99,
     'agent.polyak': 0.995,  # polyak averaging coefficient for targets update
     'agent.actor.lr': 1e-4, 
     'agent.actor.lr.scheduler': None,
     'agent.actor.min_lr': 5e-5,
     'agent.critic.lr': 1e-3,
     'agent.critic.lr.scheduler': None,
     'agent.critic.min_lr': 5e-5,
     #'agent.eps_train': 0.01,  # min eps during training
     #'agent.eps_eval': 0.001,  # eps fixed during evaluation
     #'agent.eps_decay_period': ,  # length of the epsilon decay schedule
     'agent.max_grad_norm': None,  # grad clipping by norm
     
     # only for continuous control
     'env.clip_action': True,  # clip action within valid bound before step()
     
     'replay.capacity': 1000000, 
     'replay.init_size': 10000,  # number of time steps to run random policy to initialize replay buffer
     'replay.batch_size': 100,
     
     'train.timestep': int(1e6),  # total number of training (environmental) timesteps
     'eval.freq': 5000,
     'eval.num_episode': 10
     
     
     
     
     #'agent.update_horizon': 1,
     #'agent.update_period': 4,  # period betwen DQN updates
     #'agent.target_update_period': 8000,  # period to update target network by copy latest Q-network
     
    })


def run(config, seed, device):
    set_global_seeds(seed)
    logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)
    
    def make_env():
        env = gym.make(config['env.id'])
        if config['env.time_aware_obs']:
            env = TimeAwareObservation(env)
        if config['env.clip_action'] and isinstance(env.action_space, Box):
            env = ClipAction(env)
        return env
    env = make_vec_env(make_env, 1, seed, 'serial')  # single environment
    env = VecMonitor(env)  # Note: important before standardization to get raw return
    if config['env.standardize']:  # running averages of observation and reward
        env = VecStandardizeObservation(env, clip=10.)
        
        #env = VecStandardizeReward(env, clip=10., gamma=0.99)
        
    eval_env = make_vec_env(make_env, 1, seed, 'serial')
    eval_env = VecMonitor(eval_env)
    
    agent = Agent(config, env, device)
    replay = ReplayBuffer(config['replay.capacity'], device)
    engine = Engine(config, agent=agent, env=env, eval_env=eval_env, replay=replay)
    
    train_logs, eval_logs = engine.train()
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    pickle_dump(obj=eval_logs, f=logdir/'eval_logs', ext='.pkl')
    return None  
    

if __name__ == '__main__':
    run_experiment(run=run, 
                   config=config, 
                   seeds=[4153361530], #3503522377, 2876994566, 172236777, 3949341511, 849059707], 
                   num_worker=os.cpu_count())
