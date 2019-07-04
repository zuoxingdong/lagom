import time
import os
from pathlib import Path
from itertools import count
from itertools import chain

import gym
from gym.spaces import Box
from gym.wrappers import ClipAction

import torch
import torch.multiprocessing as mp

from lagom import Logger
from lagom import EpisodeRunner
from lagom.transform import describe
from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds
from lagom.utils import color_str
from lagom.experiment import Config
from lagom.experiment import Grid
from lagom.experiment import Sample
from lagom.experiment import run_experiment
from lagom.envs import make_vec_env
from lagom.envs.wrappers import get_wrapper
from lagom.envs.wrappers import VecMonitor
from lagom.envs.wrappers import VecStandardizeObservation
from lagom.envs.wrappers import VecStandardizeReward
from lagom.envs.wrappers import VecStepInfo

from baselines.impala.agent import Agent
from baselines.impala.engine import Engine


config = Config(
    {'log.freq': 1, 
     'checkpoint.num': 3,
     
     'env.id': 'Hopper-v3',###Grid(['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'Swimmer-v3']), 
     'env.standardize_obs': False,
     'env.standardize_reward': False,
     
     'nn.sizes': [64, 64],
     
     'agent.lr': 7e-4,
     'agent.use_lr_scheduler': False,
     'agent.gamma': 0.99,
     'agent.standardize_adv': True,  # standardize advantage estimates
     'agent.max_grad_norm': 0.5,  # grad clipping by norm
     'agent.entropy_coef': 0.01,
     'agent.value_coef': 0.5,
     'agent.clip_rho': 1.0,
     'agent.clip_pg_rho': 1.0,
     'agent.num_actors': 2,
     
     # only for continuous control
     'env.clip_action': True,  # clip action within valid bound before step()
     'agent.std0': 0.6,  # initial std
     
     'train.timestep': int(1e6),  # total number of training (environmental) timesteps
     'train.timestep_per_iter': 200,
     'train.batch_size': 64,  # number of timesteps per iteration
     'eval.freq': 5000, 
     'eval.num_episode': 10
     
    })


def make_env(config, seed, mode):
    assert mode in ['train', 'eval']
    def _make_env():
        env = gym.make(config['env.id'])
        if config['env.clip_action'] and isinstance(env.action_space, Box):
            env = ClipAction(env)
        return env
    env = make_vec_env(_make_env, 1, seed)  # single environment
    env = VecMonitor(env)
    if mode == 'train':
        if config['env.standardize_obs']:
            env = VecStandardizeObservation(env, clip=5.)
        if config['env.standardize_reward']:
            env = VecStandardizeReward(env, clip=10., gamma=config['agent.gamma'])
        env = VecStepInfo(env)
    return env


def actor(config, seed, make_env, learner_agent, runner, queue):
    torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
    env = make_env(config, seed, 'train')
    agent = Agent(config, env, torch.device('cpu'))
    while learner_agent.total_timestep < config['train.timestep']:
        
        time.sleep(5.0)
        t0 = time.perf_counter()
        
        
        agent.load_state_dict(learner_agent.state_dict())  # copy to CPU by default
        with torch.no_grad():
            D = runner(agent, env, config['train.timestep_per_iter'])
            [queue.put(traj) for traj in D]
        
        print(f'Actor #{os.getpid()}: collected {len(D)} trajectories, used {round(time.perf_counter() - t0, 1)} s, Queue size: {queue.qsize()}')
        


def learner(config, logdir, agent, engine, queue):
    torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
    train_logs = []
    checkpoint_count = 0
    n = 0
    while agent.total_timestep < config['train.timestep']:
        D = []
        while len(D) < config['train.batch_size']:
            while queue.empty():
                time.sleep(0.01)
            D.append(queue.get_nowait())
        train_logger = engine.train(n, D=D)
        train_logs.append(train_logger.logs)
        if n == 0 or (n+1) % config['log.freq'] == 0:
            train_logger.dump(keys=None, index=0, indent=0, border='-'*50)
        if agent.total_timestep >= int(config['train.timestep']*(checkpoint_count/(config['checkpoint.num'] - 1))):
            agent.checkpoint(logdir, n + 1)
            checkpoint_count += 1
        n += 1
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')


def evaluator(config, logdir, seed, make_env, learner_agent):
    torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
    eval_logs = []
    env = make_env(config, seed, 'train')
    agent = Agent(config, env, torch.device('cpu'))
    runner = EpisodeRunner(reset_on_call=True)
    evaluated_steps = config['eval.freq']
    while learner_agent.total_timestep < config['train.timestep']:
        if learner_agent.total_timestep < evaluated_steps:
            time.sleep(1.0)
        else:
            t0 = time.perf_counter()
            agent.load_state_dict(learner_agent.state_dict())  # copy to CPU by default
            with torch.no_grad():
                D = []
                for _ in range(config['eval.num_episode']):
                    D += runner(agent, env, env.spec.max_episode_steps)
            logger = Logger()
            logger('num_seconds', round(time.perf_counter() - t0, 1))
            logger('num_trajectories', len(D))
            logger('num_timesteps', sum([len(traj) for traj in D]))
            logger('accumulated_trained_timesteps', learner_agent.total_timestep)

            infos = [info for info in chain.from_iterable([traj.infos for traj in D]) if 'episode' in info]
            online_returns = [info['episode']['return'] for info in infos]
            online_horizons = [info['episode']['horizon'] for info in infos]
            logger('online_return', describe(online_returns, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('online_horizon', describe(online_horizons, axis=-1, repr_indent=1, repr_prefix='\n'))

            monitor_env = get_wrapper(env, 'VecMonitor')
            logger('running_return', describe(monitor_env.return_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('running_horizon', describe(monitor_env.horizon_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger.dump(keys=None, index=0, indent=0, border=color_str('+'*50, color='green'))
            eval_logs.append(logger.logs)
            
            evaluated_steps += config['eval.freq']
    pickle_dump(obj=eval_logs, f=logdir/'eval_logs', ext='.pkl')


def run(config, seed, device, logdir):
    set_global_seeds(seed)
    
    queue = mp.Queue(maxsize=100)
    env = make_env(config, seed, 'train')
    agent = Agent(config, env, device)
    agent.share_memory()
    runner = EpisodeRunner(reset_on_call=False)
    engine = Engine(config, agent=agent, env=env, runner=runner)
    
    learner_process = mp.Process(target=learner, args=(config, logdir, agent, engine, queue))
    actor_processes = [mp.Process(target=actor, args=(config, seed, make_env, agent, runner, queue)) 
                       for _ in range(config['agent.num_actors'])]
    evaluator_process = mp.Process(target=evaluator, args=(config, logdir, seed, make_env, agent))
    
    learner_process.start()
    print('Learner started !')
    [p.start() for p in actor_processes]
    print('Actors started !')
    evaluator_process.start()
    print('Evaluator started !')
    evaluator_process.join()
    [p.join() for p in actor_processes]
    learner_process.join()
    return None
    

if __name__ == '__main__':
    mp.set_start_method('spawn')  # IMPORTANT for agent.share_memory()
    torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
    run_experiment(run=run, 
                   config=config, 
                   seeds=[1770966829],  ###[1770966829, 1500925526, 2054191100], 
                   log_dir='logs/default',
                   max_workers=None, ########os.cpu_count(), 
                   chunksize=1, 
                   use_gpu=True,  # IMPALA benefits from GPU
                   gpu_ids=None)