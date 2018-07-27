from pathlib import Path

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from lagom import set_global_seeds
from lagom import BaseAlgorithm
from lagom.envs import EnvSpec

from lagom.envs import make_envs
from lagom.envs import make_gym_env
from lagom.envs.vec_env import SerialVecEnv

from lagom.runner import SegmentRunner

from lagom.agents import A2CAgent

from engine import Engine
from policy import MLP
from policy import CategoricalPolicy


class Algorithm(BaseAlgorithm):
    def __call__(self, config):
        # Set random seeds: PyTorch, numpy.random, random
        set_global_seeds(seed=config['seed'])
        
        # Make a list of make_env functions
        list_make_env = make_envs(make_env=make_gym_env, 
                                  env_id=config['env:id'], 
                                  num_env=config['train:N'], 
                                  init_seed=config['seed']*2)
        # Create vectorized environment
        env = SerialVecEnv(list_make_env=list_make_env)
        # Create environment specification
        env_spec = EnvSpec(env)
        
        # Create device
        device = torch.device(f'cuda:{config["cuda_id"]}' if config['cuda'] else 'cpu')
        
        # Create policy
        network = MLP(config=config).to(device)
        policy = CategoricalPolicy(network=network, env_spec=env_spec)

        # Create optimizer
        optimizer = optim.Adam(policy.network.parameters(), lr=config['algo:lr'])
        # Create learning rate scheduler
        if config['algo:use_lr_scheduler']:
            max_epoch = config['train:iter']  # Max number of lr decay, Note where lr_scheduler put
            lambda_f = lambda epoch: 1 - epoch/max_epoch  # decay learning rate for each training epoch
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_f)
        
        # Create agent
        kwargs = {'device': device}
        if config['algo:use_lr_scheduler']:
            kwargs['lr_scheduler'] = lr_scheduler
        agent = A2CAgent(policy=policy, 
                         optimizer=optimizer, 
                         config=config, 
                         **kwargs)
        
        # Create runner
        runner = SegmentRunner(agent=agent, 
                               env=env, 
                               gamma=config['algo:gamma'])
        
        # Create engine
        engine = Engine(agent=agent, 
                        runner=runner, 
                        config=config, 
                        logger=None)
        
        # Training and evaluation
        train_logs = []
        eval_logs = []
        for i in range(config['train:iter']):
            train_output = engine.train(i)
            
            # Logging and evaluation
            if i == 0 or (i+1) % config['log:interval'] == 0:
                # Log training and record the loggings
                train_logger = engine.log_train(train_output)
                train_logs.append(train_logger.logs)
                # Log evaluation and record the loggings
                eval_output = engine.eval(i)
                eval_logger = engine.log_eval(eval_output)
                eval_logs.append(eval_logger.logs)

        # Save the loggings
        np.save(Path(config['log:dir']) / str(config['ID']) / 'train', train_logs)
        np.save(Path(config['log:dir']) / str(config['ID']) / 'eval', eval_logs)
        
        return None
