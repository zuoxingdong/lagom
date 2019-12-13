from multiprocessing import Pool
import time

import numpy as np
import torch
import gym
import lagom
import lagom.utils as utils

from baselines.openaies.openaies import OpenAIES
from baselines.openaies.agent import Agent


configurator = lagom.Configurator(
    {'log.freq': 10, 
     'checkpoint.inference.num': 3,
     
     'env.id': lagom.Grid(['dm2gym:CheetahRun-v0', 'dm2gym:HopperStand-v0', 'dm2gym:WalkerWalk-v0']),
     
     'nn.sizes': [64, 64],
     
     # only for continuous control
     'env.clip_action': True,  # clip action within valid bound before step()
     
     'train.generations': 300,  # total number of ES generations
     'train.popsize': 32,
     'train.worker_chunksize': 4,  # must be divisible by popsize
     'train.mu0': 0.0,
     'train.std0': 1.0,
     'train.lr': 1e-2,
     'train.lr_decay': 1.0,
     'train.min_lr': 1e-6,
     'train.sigma_scheduler_args': [1.0, 0.01, 400, 0],
     'train.antithetic': False,
     'train.rank_transform': True
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
    env = lagom.envs.TimeStepEnv(env)
    return env


def fitness(data):
    torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
    config, param = data
    env = make_env(config, 'train')
    agent = Agent(config, env)
    agent.from_vec(torch.as_tensor(param).float())
    runner = lagom.EpisodeRunner()
    with torch.no_grad():
        D = runner(agent, env, 10)
    R = np.mean([sum(traj.rewards) for traj in D])
    H = np.mean([traj.T for traj in D])
    return R, H


def run(config):
    lagom.set_global_seeds(config.seed)
    torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
    
    print('Initializing...')
    env = make_env(config, 'eval')
    agent = Agent(config, env)
    es = OpenAIES([config['train.mu0']]*agent.num_params, config['train.std0'], 
                  {'popsize': config['train.popsize'], 
                   'seed': config.seed, 
                   'sigma_scheduler_args': config['train.sigma_scheduler_args'],
                   'lr': config['train.lr'],
                   'lr_decay': config['train.lr_decay'],
                   'min_lr': config['train.min_lr'],
                   'antithetic': config['train.antithetic'],
                   'rank_transform': config['train.rank_transform']})
    
    cond_save = utils.NConditioner(max_n=config['train.generations'], num_conditions=config['checkpoint.inference.num'], mode='accumulative')
    cond_logger = utils.IntervalConditioner(interval=config['log.freq'], mode='accumulative')
    train_logs = []
    generation = 0
    if config.resume_checkpointer.exists():
        env, es, cond_save, cond_logger, train_logs, generation = lagom.checkpointer('load', config, state_obj=[agent])
        agent.env = env
    
    with Pool(processes=config['train.popsize']//config['train.worker_chunksize']) as pool:
        print('Finish initialization. Training starts...')
        while generation < config['train.generations']:
            t0 = time.perf_counter()
            solutions = es.ask()
            data = [(config, solution) for solution in solutions]
            out = pool.map(utils.CloudpickleWrapper(fitness), data, chunksize=config['train.worker_chunksize'])
            Rs, Hs = zip(*out)
            es.tell(solutions, [-R for R in Rs])
            
            logger = lagom.Logger()
            logger('generation', generation+1)
            logger('num_seconds', round(time.perf_counter() - t0, 1))
            logger('Returns', utils.describe(Rs, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('Horizons', utils.describe(Hs, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('fbest', es.result.fbest)
            train_logs.append(logger.logs)
            utils.pickle_dump(obj=train_logs, f=config.logdir/'train_logs', ext='.pkl')
            if cond_logger(generation):
                logger.dump(keys=None, index=-1, indent=0, border='-'*50)
            if cond_save(generation):
                agent.from_vec(torch.as_tensor(es.result.xbest).float())
                agent.checkpoint(config.logdir, generation+1)
            lagom.checkpointer('save', config, obj=[env, es, cond_save, cond_logger, train_logs, generation+1], state_obj=[agent])
            generation += 1
        agent.from_vec(torch.as_tensor(es.result.xbest).float())
        agent.checkpoint(config.logdir, generation+1)
    return None


if __name__ == '__main__':
    lagom.run_experiment(run=run, 
                         configurator=configurator, 
                         seeds=lagom.SEEDS[:3],
                         log_dir='logs/default',
                         max_workers=8,  # tune to fulfill computation power
                         chunksize=1, 
                         use_gpu=False,
                         gpu_ids=None)
