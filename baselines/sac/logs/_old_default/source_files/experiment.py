import os
from pathlib import Path

import gym

from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds
from lagom.experiment import Config
from lagom.experiment import Grid
from lagom.experiment import Sample
from lagom.experiment import Condition
from lagom.experiment import run_experiment
from lagom.envs import make_vec_env
from lagom.envs.wrappers import TimeLimit
from lagom.envs.wrappers import NormalizeAction
from lagom.envs.wrappers import VecMonitor

from agent import Agent
from engine import Engine
from replay_buffer import ReplayBuffer


config = Config(
    {'cuda': False, #############True, 
     'log.dir': 'logs/default', 
     'log.freq': 1000,  # every n timesteps
     'checkpoint.num': 3,
     
     'env.id': Grid(['HalfCheetah-v3']),######, 'Hopper-v3', 'Walker2d-v3', 'Swimmer-v3']),
     
     'agent.gamma': 0.99,
     'agent.polyak': 0.995,  # polyak averaging coefficient for targets update
     'agent.actor.lr': 3e-4, 
     'agent.actor.use_lr_scheduler': False,
     'agent.critic.lr': 3e-4,
     'agent.critic.use_lr_scheduler': False,
     'agent.policy_delay': 2,
     'agent.alpha0': 0.2,  # initial temperature
     'agent.max_grad_norm': 999999,  # grad clipping by norm
     
     'replay.capacity': 1000000, 
     # number of time steps to take uniform actions initially
     'replay.init_size': Condition(lambda x: 1000 if x['env.id'] in ['Hopper-v3', 'Walker2d-v3'] else 10000),
     'replay.batch_size': 256,
     
     'train.timestep': int(1e6),  # total number of training (environmental) timesteps
     'eval.freq': 5000,
     'eval.num_episode': 10
     
    })


def make_env(config, seed):
    def _make_env():
        env = gym.make(config['env.id'])
        env = env.env  # strip out gym TimeLimit, TODO: remove until gym update it
        env = TimeLimit(env, env.spec.max_episode_steps)
        env = NormalizeAction(env)
        return env
    env = make_vec_env(_make_env, 1, seed)  # single environment
    return env


def run(config, seed, device):
    set_global_seeds(seed)
    logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)
    
    env = make_env(config, seed)
    env = VecMonitor(env)
    
    eval_env = make_env(config, seed)
    eval_env = VecMonitor(eval_env)
    
    agent = Agent(config, env, device)
    replay = ReplayBuffer(env, config['replay.capacity'], device)
    engine = Engine(config, agent=agent, env=env, eval_env=eval_env, replay=replay, logdir=logdir)
    
    train_logs, eval_logs = engine.train()
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    pickle_dump(obj=eval_logs, f=logdir/'eval_logs', ext='.pkl')
    return None  
    

if __name__ == '__main__':
    run_experiment(run=run, 
                   config=config, 
                   seeds=[4153361530], #####3503522377, 2876994566, 172236777, 3949341511, 849059707], 
                   num_worker=os.cpu_count())
