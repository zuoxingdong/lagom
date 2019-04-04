import os
from pathlib import Path

import gym
from gym.spaces import Box

from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds

from lagom.experiment import Config
from lagom.experiment import Grid
from lagom.experiment import Sample
from lagom.experiment import run_experiment

from lagom.envs import make_vec_env
from lagom.envs.wrappers import TimeLimit
from lagom.envs.wrappers import TimeAwareObservation
from lagom.envs.wrappers import ClipAction
from lagom.envs.wrappers import VecMonitor

from lagom.runner import EpisodeRunner

from agent import Agent
from engine import Engine
from replay_buffer import ReplayBuffer


config = Config(
    {'cuda': True, 
     'log.dir': 'logs/default', 
     'log.freq': 1, 
     
     'env.id': Grid(['HalfCheetah-v2']),  #['Hopper-v2', 'Ant-v2']
     'env.clip_action': True,  # clip action within valid bound before step()
     'env.time_aware_obs': False,  # append time step to observation
     
     # NOTE: VecStandardizeObservation/VecStandardizeReward does NOT work well here
     # use buffer moments and popart instead
     
     'agent.gamma': 0.99,
     'agent.polyak': 0.995,  # polyak averaging coefficient for targets update
     'agent.actor.lr': 1e-4, 
     'agent.actor.use_lr_scheduler': False,
     'agent.critic.lr': 1e-3,
     'agent.critic.use_lr_scheduler': False,
     'agent.action_noise': 0.1,
     'agent.max_grad_norm': 1000,  # grad clipping by norm
     
     'replay.capacity': 1000000, 
     'replay.init_size': 10000,  # number of time steps to run random policy to initialize replay buffer
     'replay.batch_size': 128,
     
     'train.timestep': int(1e6),  # total number of training (environmental) timesteps
     'eval.freq': 5000,
     'eval.num_episode': 10
     
    })


def run(config, seed, device):
    set_global_seeds(seed)
    logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)
    
    def make_env():
        env = gym.make(config['env.id'])
        env = env.env  # strip out gym TimeLimit, TODO: remove until gym update it
        env = TimeLimit(env, env.spec.max_episode_steps)
        if config['env.time_aware_obs']:
            env = TimeAwareObservation(env)
        if config['env.clip_action'] and isinstance(env.action_space, Box):
            env = ClipAction(env)
        return env
    env = make_vec_env(make_env, 1, seed, 'serial')  # single environment
    env = VecMonitor(env)
    
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
