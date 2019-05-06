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
from lagom.envs.wrappers import ClipAction
from lagom.envs.wrappers import VecMonitor

from baselines.td3.agent import Agent
from baselines.td3.engine import Engine
from baselines.td3.replay_buffer import ReplayBuffer


config = Config(
    {'cuda': True, 
     'log.dir': 'logs/default', 
     'log.freq': 5,  # every n episodes
     'checkpoint.freq': int(1e5),  # every n timesteps
     
     'env.id': Grid(['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'Swimmer-v3']),
     
     'agent.gamma': 0.99,
     'agent.polyak': 0.995,  # polyak averaging coefficient for targets update
     'agent.actor.lr': 1e-3, 
     'agent.actor.use_lr_scheduler': False,
     'agent.critic.lr': 1e-3,
     'agent.critic.use_lr_scheduler': False,
     'agent.action_noise': 0.1,
     'agent.target_noise': 0.2,
     'agent.target_noise_clip': 0.5,
     'agent.policy_delay': 2,
     'agent.max_grad_norm': 999999,  # grad clipping by norm
     
     'replay.capacity': 1000000, 
     # number of time steps to take uniform actions initially
     'replay.init_size': Condition(lambda x: 1000 if x['env.id'] in ['Hopper-v3', 'Walker2d-v3'] else 10000),  
     'replay.batch_size': 100,
     
     'train.timestep': int(1e6),  # total number of training (environmental) timesteps
     'eval.freq': 5000,
     'eval.num_episode': 10
     
    })


def make_env(config, seed):
    def _make_env():
        env = gym.make(config['env.id'])
        env = env.env  # strip out gym TimeLimit, TODO: remove until gym update it
        env = TimeLimit(env, env.spec.max_episode_steps)
        env = ClipAction(env)
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
    replay = ReplayBuffer(config['replay.capacity'], device)
    engine = Engine(config, agent=agent, env=env, eval_env=eval_env, replay=replay, logdir=logdir)
    
    train_logs, eval_logs = engine.train()
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    pickle_dump(obj=eval_logs, f=logdir/'eval_logs', ext='.pkl')
    return None  
    

if __name__ == '__main__':
    run_experiment(run=run, 
                   config=config, 
                   seeds=[4153361530, 3503522377, 2876994566, 172236777, 3949341511, 849059707], 
                   num_worker=os.cpu_count())
