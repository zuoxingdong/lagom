import os
from pathlib import Path
from itertools import count

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
from lagom.envs.wrappers import ClipAction
from lagom.envs.wrappers import VecMonitor
from lagom.envs.wrappers import VecStandardizeObservation
from lagom.envs.wrappers import VecStandardizeReward
from lagom.runner import EpisodeRunner

from agent import Agent
from engine import Engine


config = Config(
    {'cuda': True, 
     
     ######
     'cuda_ids': [1, 2, 3, 5, 6, 7],
     ######
     
     
     
     'log.dir': 'logs/default', 
     'log.freq': 10, 
     'checkpoint.freq': 50,
     
     'env.id': Grid(['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'Swimmer-v3']), 
     'env.standardize_obs': True,
     'env.standardize_reward': True,
     
     'nn.sizes': [64, 64],
     
     'agent.lr': 1e-3,
     'agent.use_lr_scheduler': False,
     'agent.gamma': 0.99,
     'agent.gae_lambda': 0.97,
     'agent.standardize_adv': True,  # standardize advantage estimates
     'agent.max_grad_norm': 0.5,  # grad clipping by norm
     'agent.entropy_coef': 0.01,
     'agent.value_coef': 0.5,
     
     # only for continuous control
     'env.clip_action': True,  # clip action within valid bound before step()
     'agent.std0': 0.5,  # initial std
     'agent.std_style': 'exp',  # std parameterization
     'agent.std_range': None,  # bounded std: (min, max)
     'agent.beta': None,  # beta-sigmoidal
     
     'train.timestep': int(1e6),  # total number of training (environmental) timesteps
     'train.timestep_per_iter': 1000,  # number of timesteps per iteration
     
    })


def make_env(config, seed):
    def _make_env():
        env = gym.make(config['env.id'])
        env = env.env  # strip out gym TimeLimit, TODO: remove until gym update it
        env = TimeLimit(env, env.spec.max_episode_steps)
        if config['env.clip_action'] and isinstance(env.action_space, Box):
            env = ClipAction(env)
        return env
    env = make_vec_env(_make_env, 1, seed)  # single environment
    return env
    

def run(config, seed, device):
    set_global_seeds(seed)
    logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)
    
    env = make_env(config, seed)
    env = VecMonitor(env)
    if config['env.standardize_obs']:
        env = VecStandardizeObservation(env, clip=5.)
    if config['env.standardize_reward']:
        env = VecStandardizeReward(env, clip=10., gamma=config['agent.gamma'])
    
    agent = Agent(config, env, device)
    runner = EpisodeRunner(reset_on_call=False)
    engine = Engine(config, agent=agent, env=env, runner=runner)
    train_logs = []
    for i in count():
        if agent.total_timestep >= config['train.timestep']:
            break
        train_logger = engine.train(i)
        train_logs.append(train_logger.logs)
        if i == 0 or (i+1) % config['log.freq'] == 0:
            train_logger.dump(keys=None, index=0, indent=0, border='-'*50)
        if i == 0 or (i+1) % config['checkpoint.freq'] == 0:
            agent.checkpoint(logdir, i + 1)
    agent.checkpoint(logdir, i + 1)
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    return None
    

if __name__ == '__main__':
    run_experiment(run=run, 
                   config=config, 
                   seeds=[1770966829, 1500925526, 2054191100], 
                   num_worker=os.cpu_count())
