from copy import deepcopy

import numpy as np

import torch
import torch.optim as optim

from lagom import BaseAlgorithm
from lagom.agents import A2CAgent
from lagom.core.policies import CategoricalMLPPolicy
from lagom.core.utils import Logger
from lagom.runner import Runner
from lagom.envs import EnvSpec

from lagom.utils import set_global_seeds

from engine import Engine


class Algorithm(BaseAlgorithm):
    def run(self, config):
        # Take out the environment
        env = config['env']
        
        # Set random seed, e.g. PyTorch, environment, numpy
        env.seed(config['seed'])
        set_global_seeds(config['seed'])
        
        # Create a logger with algorithm name
        logger = Logger(name=f'{self.name}')
        # Create environment specification
        env_spec = EnvSpec(env)
        # Create a goal-conditional policy
        policy = CategoricalMLPPolicy(env_spec, config)
        # Create an optimzer
        optimizer = optim.RMSprop(policy.parameters(), lr=config['lr'], alpha=0.99, eps=1e-5)
        # Learning rate scheduler
        max_epoch = config['train_iter']  # Max number of lr decay, Note where lr_scheduler put
        lambda_f = lambda epoch: 1 - epoch/max_epoch  # decay learning rate for each training epoch
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_f)
        # Create an agent
        agent = A2CAgent(policy, optimizer, lr_scheduler, config)
        # Create a Runner
        runner = Runner(agent, env, config['gamma'])
        # Create an engine (training and evaluation)
        engine = Engine(agent, runner, config, logger)
            
        # Create a goal sampler
        for iter_num in range(config['train_iter']):
            engine.train(iter_num)
            
        # Save the logger
        logger.save(name=f'{self.name}_ID_{config["ID"]}')
            