from multiprocessing import Pool
import time

import numpy as np
import torch
import gym
import lagom
from lagom import Logger
from lagom import EpisodeRunner
from lagom.transform import describe
from lagom.utils import CloudpickleWrapper
from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds
from lagom.utils import NConditioner, IntervalConditioner
from lagom.experiment import Config
from lagom.experiment import Configurator
from lagom.experiment import Grid
from lagom.experiment import run_experiment
from lagom.experiment import checkpointer
from lagom.envs import TimeStepEnv

from lagom import CEM
from baselines.cem.agent import Agent


configurator = Configurator(
    {'log.freq': 10, 
     'checkpoint.inference.num': 3,
     
     'env.id': Grid(['dm2gym:CheetahRun-v0', 'dm2gym:HopperStand-v0', 'dm2gym:WalkerWalk-v0']), 
     
     'nn.sizes': [64, 64],
     
     # only for continuous control
     'env.clip_action': True,  # clip action within valid bound before step()
     
     'train.generations': 300,  # total number of ES generations
     'train.popsize': 32,
     'train.worker_chunksize': 4,  # must be divisible by popsize
     'train.mu0': 0.0,
     'train.std0': 1.0,
     'train.elite_ratio': 0.2,
     'train.noise_scheduler_args': [0.01, 0.001, 300, 0]  # [initial, final, N, start]
    }, 
    num_sample=1)


def make_env(config, mode):
    assert mode in ['train', 'eval']
    env = gym.make(config['env.id'])
    env = gym.wrappers.FlattenObservation(env)
    env.seed(config.seed)
    env.observation_space.seed(config.seed)
    env.action_space.seed(config.seed)
    if config['env.clip_action'] and isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)  # TODO: use tanh to squash policy output when RescaleAction wrapper merged in gym
    env = TimeStepEnv(env)
    return env


def fitness(data):
    torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
    config, param = data
    env = make_env(config, 'train')
    agent = Agent(config, env)
    agent.from_vec(torch.as_tensor(param).float())
    runner = EpisodeRunner()
    with torch.no_grad():
        D = runner(agent, env, 10)
    R = np.mean([sum(traj.rewards) for traj in D])
    H = np.mean([traj.T for traj in D])
    return R, H


def run(config):
    set_global_seeds(config.seed)
    torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
    
    print('Initializing...')
    env = make_env(config, 'eval')
    agent = Agent(config, env)
    es = CEM([config['train.mu0']]*agent.num_params, config['train.std0'], 
             {'popsize': config['train.popsize'], 
              'seed': config.seed, 
              'elite_ratio': config['train.elite_ratio'], 
              'noise_scheduler_args': config['train.noise_scheduler_args']})
    
    cond_save = NConditioner(max_n=config['train.generations'], num_conditions=config['checkpoint.inference.num'], mode='accumulative')
    train_logs = []
    generation = 0
    if config.resume_checkpointer.exists():
        out = checkpointer('load', config, env=env, agent=agent, es=es, cond_save=cond_save, train_logs=train_logs, generation=generation)
        env = out['env']
        es = out['es']
        cond_save = out['cond_save']
        train_logs = out['train_logs']
        generation = out['generation']
        agent.env = env
    
    with Pool(processes=config['train.popsize']//config['train.worker_chunksize']) as pool:
        print('Finish initialization. Training starts...')
        while generation < config['train.generations']:
            t0 = time.perf_counter()
            solutions = es.ask()
            data = [(config, solution) for solution in solutions]
            out = pool.map(CloudpickleWrapper(fitness), data, chunksize=config['train.worker_chunksize'])
            Rs, Hs = zip(*out)
            es.tell(solutions, [-R for R in Rs])
            
            logger = Logger()
            logger('generation', generation+1)
            logger('num_seconds', round(time.perf_counter() - t0, 1))
            logger('Returns', describe(Rs, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('Horizons', describe(Hs, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('fbest', es.result.fbest)
            train_logs.append(logger.logs)
            pickle_dump(obj=train_logs, f=config.logdir/'train_logs', ext='.pkl')
            cond = IntervalConditioner(interval=config['log.freq'], mode='accumulative')
            if cond(generation):
                logger.dump(keys=None, index=-1, indent=0, border='-'*50)
            if cond_save(generation):
                agent.from_vec(torch.as_tensor(es.result.xbest).float())
                agent.checkpoint(config.logdir, generation+1)
            checkpointer('save', config, env=env, agent=agent, es=es, cond_save=cond_save, train_logs=train_logs, generation=generation+1)
            generation += 1
        agent.from_vec(torch.as_tensor(es.result.xbest).float())
        agent.checkpoint(config.logdir, generation+1)
    return None


if __name__ == '__main__':
    run_experiment(run=run, 
                   configurator=configurator, 
                   seeds=lagom.SEEDS[:3],
                   log_dir='logs/default',
                   max_workers=8,  # tune to fulfill computation power
                   chunksize=1, 
                   use_gpu=False,
                   gpu_ids=None)

