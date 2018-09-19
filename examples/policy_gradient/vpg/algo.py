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
from lagom.envs.vec_env import VecStandardize

from lagom.core.policies import CategoricalPolicy
from lagom.core.policies import GaussianPolicy

from lagom.runner import TrajectoryRunner

from lagom.agents import VPGAgent

from engine import Engine
from policy import Network


class Algorithm(BaseAlgorithm):
    def __call__(self, config, seed, device_str):
        # Set random seeds
        set_global_seeds(seed)
        # Create device
        device = torch.device(device_str)
        # Use log dir for current job (run_experiment)
        logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)
        
        # Make environment (VecEnv) for training and evaluating
        env = make_vec_env(vec_env_class=SerialVecEnv, 
                           make_env=make_gym_env, 
                           env_id=config['env.id'], 
                           num_env=1, 
                           init_seed=seed)
        eval_env = make_vec_env(vec_env_class=SerialVecEnv, 
                                make_env=make_gym_env, 
                                env_id=config['env.id'], 
                                num_env=1, 
                                init_seed=seed)
        if config['env.standardize']:  # wrap with VecStandardize for running averages of observation and rewards
            env = VecStandardize(venv=env, 
                                 use_obs=True, 
                                 use_reward=True, 
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
        
        # Create policy
        network = Network(config=config, env_spec=env_spec)
        if env_spec.control_type == 'Discrete':
            policy = CategoricalPolicy(config=config, network=network, env_spec=env_spec, learn_V=True)
        elif env_spec.control_type == 'Continuous':
            policy = GaussianPolicy(config=config, 
                                    network=network, 
                                    env_spec=env_spec, 
                                    learn_V=True,
                                    min_std=config['agent.min_std'], 
                                    std_style=config['agent.std_style'], 
                                    constant_std=config['agent.constant_std'],
                                    std_state_dependent=config['agent.std_state_dependent'],
                                    init_std=config['agent.init_std'])
        network = network.to(device)
        
        # Create optimizer and learning rate scheduler
        optimizer = optim.Adam(policy.network.parameters(), lr=config['algo.lr'])
        if config['algo.use_lr_scheduler']:
            if 'train.iter' in config:  # iteration-based training
                max_epoch = config['train.iter']
            elif 'train.timestep' in config:  # timestep-based training
                max_epoch = config['train.timestep'] + 1  # +1 to avoid 0.0 lr in final iteration
            lambda_f = lambda epoch: 1 - epoch/max_epoch  # decay learning rate for each training epoch
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_f)
        
        # Create agent
        kwargs = {'device': device}
        if config['algo.use_lr_scheduler']:
            kwargs['lr_scheduler'] = lr_scheduler
        agent = VPGAgent(config=config, 
                         policy=policy, 
                         optimizer=optimizer, 
                         **kwargs)
        
        # Create runner
        runner = TrajectoryRunner(agent=agent, 
                                  env=env, 
                                  gamma=config['algo.gamma'])
        eval_runner = TrajectoryRunner(agent=agent, 
                                       env=eval_env, 
                                       gamma=1.0)
        
        # Create engine
        engine = Engine(agent=agent, 
                        runner=runner, 
                        config=config, 
                        eval_runner=eval_runner)
        
        # Training and evaluation
        train_logs = []
        eval_logs = []
        
        for i in count():  # incremental iteration
            if 'train.iter' in config and i >= config['train.iter']:  # enough iterations
                break
            elif 'train.timestep' in config and agent.total_T >= config['train.timestep']:  # enough timesteps
                break
            
            # train and evaluation
            train_output = engine.train(n=i)
            
            # logging
            if i == 0 or (i+1) % config['log.record_interval'] == 0 or (i+1) % config['log.print_interval'] == 0:
                train_log = engine.log_train(train_output)
                
                with torch.no_grad():  # disable grad, save memory
                    eval_output = engine.eval(n=i)
                eval_log = engine.log_eval(eval_output)
                
                if i == 0 or (i+1) % config['log.record_interval'] == 0:  # record loggings
                    train_logs.append(train_log)
                    eval_logs.append(eval_log)
        
        # Save all loggings
        pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
        pickle_dump(obj=eval_logs, f=logdir/'eval_logs', ext='.pkl')
        
        return None
