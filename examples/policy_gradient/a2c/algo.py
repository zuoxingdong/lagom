from pathlib import Path

from itertools import count

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lagom import set_global_seeds
from lagom import BaseAlgorithm
from lagom import pickle_dump

from lagom.envs import make_gym_env
from lagom.envs import make_vec_env
from lagom.envs import EnvSpec
from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.vec_env import ParallelVecEnv
from lagom.envs.vec_env import VecStandardize

from lagom.core.policies import CategoricalPolicy
from lagom.core.policies import GaussianPolicy

from lagom.runner import TrajectoryRunner
from lagom.runner import SegmentRunner

from lagom.agents import A2CAgent

from engine import Engine
from policy import Network
from policy import LSTM


class Algorithm(BaseAlgorithm):
    def __call__(self, config, seed, device_str):
        set_global_seeds(seed)
        device = torch.device(device_str)
        logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)
        
        # Environment related
        env = make_vec_env(vec_env_class=SerialVecEnv, 
                           make_env=make_gym_env, 
                           env_id=config['env.id'], 
                           num_env=config['train.N'],  # batched environment
                           init_seed=seed, 
                           rolling=True)
        eval_env = make_vec_env(vec_env_class=SerialVecEnv, 
                                make_env=make_gym_env, 
                                env_id=config['env.id'], 
                                num_env=config['eval.N'], 
                                init_seed=seed, 
                                rolling=False)
        if config['env.standardize']:  # running averages of observation and reward
            env = VecStandardize(venv=env, 
                                 use_obs=True, 
                                 use_reward=False,  # A2C
                                 clip_obs=10., 
                                 clip_reward=10., 
                                 gamma=0.99, 
                                 eps=1e-8)
            eval_env = VecStandardize(venv=eval_env,  # remember to synchronize running averages during evaluation !!!
                                      use_obs=True, 
                                      use_reward=False,  # do not process rewards, no training
                                      clip_obs=env.clip_obs, 
                                      clip_reward=env.clip_reward, 
                                      gamma=env.gamma, 
                                      eps=env.eps, 
                                      constant_obs_mean=env.obs_runningavg.mu,  # use current running average as constant
                                      constant_obs_std=env.obs_runningavg.sigma)
        env_spec = EnvSpec(env)
        
        # Network and policy
        if config['network.recurrent']:
            network = LSTM(config=config, device=device, env_spec=env_spec)
        else:
            network = Network(config=config, device=device, env_spec=env_spec)
        if env_spec.control_type == 'Discrete':
            policy = CategoricalPolicy(config=config, 
                                       network=network, 
                                       env_spec=env_spec, 
                                       device=device,
                                       learn_V=True)
        elif env_spec.control_type == 'Continuous':
            policy = GaussianPolicy(config=config, 
                                    network=network, 
                                    env_spec=env_spec, 
                                    device=device,
                                    learn_V=True,
                                    min_std=config['agent.min_std'], 
                                    std_style=config['agent.std_style'], 
                                    constant_std=config['agent.constant_std'],
                                    std_state_dependent=config['agent.std_state_dependent'],
                                    init_std=config['agent.init_std'])
        
        # Optimizer and learning rate scheduler
        optimizer = optim.Adam(policy.network.parameters(), lr=config['algo.lr'])
        if config['algo.use_lr_scheduler']:
            if 'train.iter' in config:  # iteration-based
                max_epoch = config['train.iter']
            elif 'train.timestep' in config:  # timestep-based
                max_epoch = config['train.timestep'] + 1  # avoid zero lr in final iteration
            lambda_f = lambda epoch: 1 - epoch/max_epoch
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_f)
        
        # Agent
        kwargs = {'device': device}
        if config['algo.use_lr_scheduler']:
            kwargs['lr_scheduler'] = lr_scheduler
        agent = A2CAgent(config=config, 
                         policy=policy, 
                         optimizer=optimizer, 
                         **kwargs)
        
        # Runner
        runner = SegmentRunner(agent=agent, 
                               env=env, 
                               gamma=config['algo.gamma'])
        eval_runner = TrajectoryRunner(agent=agent, 
                                       env=eval_env, 
                                       gamma=1.0)
        
        # Engine
        engine = Engine(agent=agent, 
                        runner=runner, 
                        config=config, 
                        eval_runner=eval_runner)
        
        # Training and evaluation
        train_logs = []
        eval_logs = []
        
        if config['network.recurrent']:
            rnn_states_buffer = agent.policy.rnn_states  # for SegmentRunner
        
        for i in count():
            if 'train.iter' in config and i >= config['train.iter']:  # enough iterations
                break
            elif 'train.timestep' in config and agent.total_T >= config['train.timestep']:  # enough timesteps
                break
            
            if config['network.recurrent']:
                if isinstance(rnn_states_buffer, list):  # LSTM: [h, c]
                    rnn_states_buffer = [buf.detach() for buf in rnn_states_buffer]
                else:
                    rnn_states_buffer = rnn_states_buffer.detach()
                agent.policy.rnn_states = rnn_states_buffer
                
            train_output = engine.train(n=i)
            
            # Logging
            if i == 0 or (i+1) % config['log.record_interval'] == 0 or (i+1) % config['log.print_interval'] == 0:
                train_log = engine.log_train(train_output)
                
                if config['network.recurrent']:
                    rnn_states_buffer = agent.policy.rnn_states  # for SegmentRunner
                    
                with torch.no_grad():  # disable grad, save memory
                    eval_output = engine.eval(n=i)
                eval_log = engine.log_eval(eval_output)
                
                if i == 0 or (i+1) % config['log.record_interval'] == 0:
                    train_logs.append(train_log)
                    eval_logs.append(eval_log)
        
        # Save all loggings
        pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
        pickle_dump(obj=eval_logs, f=logdir/'eval_logs', ext='.pkl')
        
        return None
