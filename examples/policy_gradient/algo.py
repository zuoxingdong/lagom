import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from lagom import set_global_seeds
from lagom import BaseAlgorithm
from lagom.envs import EnvSpec
from lagom.core.utils import Logger

from lagom.agents import REINFORCEAgent
from lagom.agents import ActorCriticAgent

from lagom.runner import Runner

from engine import Engine
from policy import MLP, CategoricalPolicy
from utils import make_env


class Algorithm(BaseAlgorithm):
    def __call__(self, config):
        # Set random seeds: PyTorch, numpy.random, random
        set_global_seeds(seed=config['seed'])
        
        # Create environment and seed it
        env = make_env(seed=config['seed'], 
                       monitor=False, 
                       monitor_dir=None)
        # Create environment specification
        env_spec = EnvSpec(env)  # TODO: integrate within make_env globally
        
        # Create device
        device = torch.device('cuda' if config['cuda'] else 'cpu')
        
        # Create logger
        logger = Logger(name='logger')
        
        # Create policy
        network = MLP(config=config)
        policy = CategoricalPolicy(network=network, env_spec=env_spec)
        policy.network = policy.network.to(device)

        # Create optimizer
        optimizer = optim.Adam(policy.network.parameters(), lr=config['lr'])
        # Learning rate scheduler
        max_epoch = config['train_iter']  # Max number of lr decay, Note where lr_scheduler put
        lambda_f = lambda epoch: 1 - epoch/max_epoch  # decay learning rate for each training epoch
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_f)
        
        # Create agent
        agent_class = ActorCriticAgent
        agent = agent_class(policy=policy, 
                            optimizer=optimizer, 
                            config=config, 
                            lr_scheduler=lr_scheduler, 
                            device=device)
        
        # Create runner
        runner = Runner(agent=agent, 
                        env=env, 
                        gamma=config['gamma'])
        
        # Create engine
        engine = Engine(agent=agent, 
                        runner=runner, 
                        config=config, 
                        logger=logger)
        
        # Training
        engine.train()
        
        return None