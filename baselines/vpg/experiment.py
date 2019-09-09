import os
from itertools import count
import gym

from lagom import StepRunner
from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds
from lagom.experiment import Config
from lagom.experiment import Grid
from lagom.experiment import run_experiment
from lagom.envs import RecordEpisodeStatistics
from lagom.envs import NormalizeObservation
from lagom.envs import NormalizeReward
from lagom.envs import TimeStepEnv

from baselines.vpg.agent import Agent
from baselines.vpg.agent_lstm import Agent as LSTMAgent
from baselines.vpg.engine import Engine


config = Config(
    {'log.freq': 10, 
     'checkpoint.num': 3,
     
     'env.id': Grid(['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3']), 
     'env.normalize_obs': True,
     'env.normalize_reward': True,
     
     'use_lstm': Grid([True, False]),
     'rnn.size': 128,
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
     'agent.std0': 0.6,  # initial std
     
     'train.timestep': int(1e6),  # total number of training (environmental) timesteps
     'train.timestep_per_iter': 1000,  # number of timesteps per iteration
    })


def make_env(config, seed, mode):
    assert mode in ['train', 'eval']
    env = gym.make(config['env.id'])
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
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
    

def run(config, seed, device, logdir):
    set_global_seeds(seed)
    
    env = make_env(config, seed, 'train')
    if config['use_lstm']:
        agent = LSTMAgent(config, env, device)
    else:
        agent = Agent(config, env, device)
    runner = StepRunner(reset_on_call=False)
    engine = Engine(config, agent=agent, env=env, runner=runner)
    train_logs = []
    checkpoint_count = 0
    for i in count():
        if agent.total_timestep >= config['train.timestep']:
            break
        train_logger = engine.train(i)
        train_logs.append(train_logger.logs)
        if i == 0 or (i+1) % config['log.freq'] == 0:
            train_logger.dump(keys=None, index=0, indent=0, border='-'*50)
        if agent.total_timestep >= int(config['train.timestep']*(checkpoint_count/(config['checkpoint.num'] - 1))):
            agent.checkpoint(logdir, i + 1)
            checkpoint_count += 1
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    return None
    

if __name__ == '__main__':
    run_experiment(run=run, 
                   config=config, 
                   seeds=[1770966829, 1500925526, 2054191100], 
                   log_dir='logs/default',
                   max_workers=os.cpu_count(), 
                   chunksize=1, 
                   use_gpu=False,  # CPU a bit faster
                   gpu_ids=None)
