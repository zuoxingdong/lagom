from multiprocessing import Pool
import time

import numpy as np
import torch
import gym
import gym.wrappers
import lagom
import lagom.utils as utils

from baselines.cmaes.agent import Agent


configurator = lagom.Configurator(
    {'log.freq': 10, 
     'checkpoint.agent.num': 3,
     'checkpoint.resume.num': 5,
     
     'env.id': lagom.Grid(['CartPole-v1', 'Pendulum-v0']),
     'nn.sizes': [32, 32],
     
     'train.generations': 300,  # total number of ES generations
     'train.popsize': 32,
     'train.worker_chunksize': 4,  # must be divisible by popsize
     'train.mu0': 0.0,
     'train.std0': 1.0,
    }, 
    num_sample=1)


def make_env(config, mode):
    assert mode in ['train', 'eval']
    env = gym.make(config['env.id'])
    env.seed(config.seed)
    env.observation_space.seed(config.seed)
    env.action_space.seed(config.seed)
    if isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)  # TODO: use tanh to squash policy output when RescaleAction wrapper merged in gym
    env = lagom.envs.TimeStepEnv(env)
    return env


def fitness(data):
    config, param = data
    env = make_env(config, 'eval')
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
    es = lagom.CMAES([config['train.mu0']]*agent.num_params, config['train.std0'], 
                     {'popsize': config['train.popsize'], 
                      'seed': config.seed})
    
    cond_agent = utils.Conditioner(stop=config['train.generations'], step=config['train.generations']//config['checkpoint.agent.num'])
    cond_resume = utils.Conditioner(stop=config['train.generations'], step=config['train.generations']//config['checkpoint.resume.num'])
    cond_log = utils.Conditioner(step=config['log.freq'])
    train_logs = []
    generation = 0
    if config.resume_checkpointer.exists():
        env, es, cond_agent, cond_resume, cond_log, train_logs, generation = lagom.checkpointer('load', config, state_obj=[agent])
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
            describe_it = lambda x: utils.describe(x, axis=-1, repr_indent=1, repr_prefix='\n')
            logger('Returns', describe_it(Rs))
            logger('Horizons', describe_it(Hs))
            logger('fbest', es.result.fbest)
            train_logs.append(logger.logs)
            if cond_log(generation):
                logger.dump(keys=None, index=-1, indent=0, border='-'*50)
            if cond_agent(generation):
                agent.from_vec(torch.as_tensor(es.result.xbest).float())
                agent.checkpoint(config.logdir, generation+1)
            if cond_resume(generation):
                utils.pickle_dump(obj=train_logs, f=config.logdir/'train_logs', ext='.pkl')
                lagom.checkpointer('save', config, obj=[env, es, cond_agent, cond_resume, cond_log, train_logs, generation+1], state_obj=[agent])
            generation += 1
        utils.pickle_dump(obj=train_logs, f=config.logdir/'train_logs', ext='.pkl')
        agent.from_vec(torch.as_tensor(es.result.xbest).float())
        agent.checkpoint(config.logdir, generation+1)
    return None


if __name__ == '__main__':
    lagom.run_experiment(run=run, 
                         configurator=configurator, 
                         seeds=lagom.SEEDS[:3],
                         log_dir='logs/default',
                         max_workers=8,  # fulfill CPU: total cores / processes
                         chunksize=1, 
                         use_gpu=False,
                         gpu_ids=None)
