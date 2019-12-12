import os
import gym

import lagom
from lagom import StepRunner
from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds
from lagom.utils import NConditioner
from lagom.experiment import Config
from lagom.experiment import Configurator
from lagom.experiment import Grid
from lagom.experiment import run_experiment
from lagom.experiment import checkpointer
from lagom.envs import RecordEpisodeStatistics
from lagom.envs import NormalizeObservation
from lagom.envs import NormalizeReward
from lagom.envs import TimeStepEnv

from baselines.ppo.agent import Agent
from baselines.ppo.engine import Engine


configurator = Configurator(
    {'log.freq': 10, 
     'checkpoint.inference.num': 3,
     
     'env.id': Grid(['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3']),
     'env.normalize_obs': True,
     'env.normalize_reward': True,
     
     'nn.sizes': [64, 64],
     
     'agent.policy_lr': 3e-4,
     'agent.use_lr_scheduler': True,
     'agent.value_lr': 1e-3,
     'agent.gamma': 0.99,
     'agent.gae_lambda': 0.95,
     'agent.standardize_adv': True,  # standardize advantage estimates
     'agent.max_grad_norm': 0.5,  # grad clipping by norm
     'agent.clip_range': 0.2,  # ratio clipping
     
     # only for continuous control
     'env.clip_action': True,  # clip action within valid bound before step()
     'agent.std0': 0.4,  # initial std
     
     'train.timestep': int(1e6),  # total number of training (environmental) timesteps
     'train.timestep_per_iter': 2048,  # number of timesteps per iteration
     'train.batch_size': 64,
     'train.num_epochs': 10,
    }, 
    num_sample=1)


def make_env(config, mode):
    assert mode in ['train', 'eval']
    env = gym.make(config['env.id'])
    env.seed(config.seed)
    env.observation_space.seed(config.seed)
    env.action_space.seed(config.seed)
    if config['env.clip_action'] and isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)
    if mode == 'train':
        env = RecordEpisodeStatistics(env, deque_size=100)
        if config['env.normalize_obs']:
            env = NormalizeObservation(env, clip=5.)
        if config['env.normalize_reward']:
            env = NormalizeReward(env, clip=10., gamma=config['agent.gamma'])
    env = TimeStepEnv(env)
    return env
    

def run(config):
    set_global_seeds(config.seed)
    
    env = make_env(config, 'train')
    agent = Agent(config, env)
    runner = StepRunner(reset_on_call=False)
    engine = Engine(config, agent=agent, env=env, runner=runner)
    
    cond_save = NConditioner(max_n=config['train.timestep'], num_conditions=config['checkpoint.inference.num'], mode='accumulative')
    train_logs = []
    iteration = 0
    if config.resume_checkpointer.exists():
        out = checkpointer('load', config, env=env, agent=agent, policy_optimizer=agent.policy_optimizer, value_optimizer=agent.value_optimizer, runner=runner, train_logs=train_logs, iteration=iteration)
        env = out['env']
        runner = out['runner']
        train_logs = out['train_logs']
        iteration = out['iteration']
        agent.env = env
        engine.env = env
        engine.runner = runner

    while agent.total_timestep < config['train.timestep']:
        train_logger = engine.train(iteration)
        train_logs.append(train_logger.logs)
        if cond_save(int(agent.total_timestep)):
            agent.checkpoint(config.logdir, iteration+1)
        pickle_dump(obj=train_logs, f=config.logdir/'train_logs', ext='.pkl')
        checkpointer('save', config, env=env, agent=agent, policy_optimizer=agent.policy_optimizer, value_optimizer=agent.value_optimizer, runner=runner, train_logs=train_logs, iteration=iteration+1)
        iteration += 1
    agent.checkpoint(config.logdir, iteration+1)
    return None
    

if __name__ == '__main__':
    run_experiment(run=run,
                   configurator=configurator,
                   seeds=lagom.SEEDS[:3],
                   log_dir='logs/default',
                   max_workers=os.cpu_count(),
                   chunksize=1,
                   use_gpu=False,  # CPU a bit faster
                   gpu_ids=None)
