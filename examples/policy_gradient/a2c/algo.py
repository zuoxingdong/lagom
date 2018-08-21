from pathlib import Path

from itertools import count

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
from lagom.envs.vec_env import StandardizeVecEnv

from lagom.runner import SegmentRunner

from lagom.agents import A2CAgent

from engine import Engine
from policy import CategoricalMLP
from policy import CategoricalPolicy
from policy import GaussianMLP
from policy import GaussianPolicy


class Algorithm(BaseAlgorithm):
    def __call__(self, config):
        # Set random seeds: PyTorch, numpy.random, random
        set_global_seeds(seed=config['seed'])
        
        # Create an VecEnv environment
        list_make_env = make_envs(make_env=make_gym_env, 
                                  env_id=config['env:id'], 
                                  num_env=config['train:N'], 
                                  init_seed=config['seed'])
        env = SerialVecEnv(list_make_env)
        # Wrapper to standardize observation and reward from running average
        if config['env:normalize']:
            env = StandardizeVecEnv(venv=env, 
                                    use_obs=True, 
                                    use_reward=True, 
                                    clip_obs=10., 
                                    clip_reward=10., 
                                    gamma=0.99, 
                                    eps=1e-8)
        # Create environment specification
        env_spec = EnvSpec(env)
        
        # Create device object, note that in BaseExperimentWorker already assigns a specific GPU for this task
        device = torch.device(f'cuda:{torch.cuda.current_device()}' if config['cuda'] else 'cpu')
        
        # Create policy
        if env_spec.control_type == 'Discrete':
            network = CategoricalMLP(config=config, env_spec=env_spec).to(device)
            policy = CategoricalPolicy(network=network, 
                                       env_spec=env_spec, 
                                       config=config)
        elif env_spec.control_type == 'Continuous':
            network = GaussianMLP(config=config, env_spec=env_spec).to(device)
            policy = GaussianPolicy(network=network, 
                                    env_spec=env_spec, 
                                    config=config,
                                    min_std=config['agent:min_std'], 
                                    std_style=config['agent:std_style'], 
                                    constant_std=config['agent:constant_std'])

        # Create optimizer
        optimizer = optim.Adam(policy.network.parameters(), lr=config['algo:lr'])
        # Create learning rate scheduler
        if config['algo:use_lr_scheduler']:
            # Define max number of lr decay
            if 'train:iter' in config:  # iteration-based training
                max_epoch = config['train:iter']
            elif 'train:timestep' in config:  # timestep-based training
                max_epoch = config['train:timestep'] + 1  # plus 1 avoid having 0.0 lr in final iteration
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
        
        for i in count():  # successively increment iteration
            # Terminate until condition is met
            if 'train:iter' in config and i >= config['train:iter']:  # enough iteration, terminate
                break
            elif 'train:timestep' in config and agent.accumulated_trained_timesteps >= config['train:timestep']:
                break
                
            # Do training
            train_output = engine.train(i)
            
            # Logging and evaluation
            if i == 0 or (i+1) % config['log:interval'] == 0:
                # Log training and record the loggings
                train_logger = engine.log_train(train_output)
                train_logs.append(train_logger.logs)
                # Log evaluation and record the loggings
                with torch.no_grad():  # no need to have gradient, save memory
                    eval_output = engine.eval(i)
                    eval_logger = engine.log_eval(eval_output)
                    eval_logs.append(eval_logger.logs)
                    
                # Save the logging periodically
                # This is good to avoid saving very large file at once, because the program might get stuck
                # The file name is augmented with current iteration
                np.save(Path(config['log:dir']) / str(config['ID']) / f'train:{i}', train_logs)
                np.save(Path(config['log:dir']) / str(config['ID']) / f'eval:{i}', eval_logs)
                # Clear the logging list
                train_logs.clear()
                eval_logs.clear()
        
        return None
