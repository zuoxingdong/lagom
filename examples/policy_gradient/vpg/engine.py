import numpy as np

import torch

from lagom import Logger
from lagom import color_str

from lagom.engine import BaseEngine

from lagom.envs import make_gym_env
from lagom.envs import make_envs
from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.vec_env import VecStandardize

from lagom.runner import TrajectoryRunner


class Engine(BaseEngine):
    def train(self, n):
        self.agent.policy.network.train()  # set to train mode
        
        # Collect a list of trajectories
        D = self.runner(N=self.config['train.N'], T=self.config['train.T'])
        
        # Train agent with collected data
        out_agent = self.agent.learn(D)
        # Return training output
        train_output = {}
        train_output['D'] = D
        train_output['out_agent'] = out_agent
        train_output['n'] = n
        
        return train_output
        
    def log_train(self, train_output, **kwargs):
        # Create training logger
        logger = Logger(name='train_logger')
        
        # Unpack training output for logging
        D = train_output['D']
        out_agent = train_output['out_agent']
        n = train_output['n']
        
        # Loggings: use item() to save memory
        logger.log('train_iteration', n+1)  # iteration starts from 1
        if self.config['algo.use_lr_scheduler']:
            logger.log('current_lr', out_agent['current_lr'])

        logger.log('loss', out_agent['loss'])
        logger.log('policy_loss', out_agent['policy_loss'])
        logger.log('policy_entropy', -out_agent['entropy_loss'])  # negate entropy loss is entropy
        logger.log('value_loss', out_agent['value_loss'])
        
        # Log something about trajectories
        batch_returns = [sum(trajectory.all_r) for trajectory in D]
        batch_discounted_returns = [trajectory.all_discounted_returns[0] for trajectory in D]
        num_timesteps = sum([trajectory.T for trajectory in D])
        
        logger.log('num_trajectories', len(D))
        logger.log('num_timesteps', num_timesteps)
        logger.log('accumulated_trained_timesteps', self.agent.total_T)
        logger.log('average_return', np.mean(batch_returns))
        logger.log('average_discounted_return', np.mean(batch_discounted_returns))
        logger.log('std_return', np.std(batch_returns))
        logger.log('min_return', np.min(batch_returns))
        logger.log('max_return', np.max(batch_returns))

        # Dump loggings
        if n == 0 or (n+1) % self.config['log.print_interval'] == 0:
            print('-'*50)
            logger.dump(keys=None, index=None, indent=0)
            print('-'*50)

        return logger.logs
        
    def eval(self, n):
        self.agent.policy.network.eval()  # set to evaluation mode
        
        # synchronize running average if environment wrapped by VecStandardize
        if self.config['env.standardize']:
            self.eval_runner.env.constant_obs_mean = self.runner.env.obs_runningavg.mu
            self.eval_runner.env.constant_obs_std = self.runner.env.obs_runningavg.sigma
        
        # Collect a list of trajectories
        T = self.eval_runner.env.T
        D = self.eval_runner(N=self.config['eval.N'], T=T)
        
        # Return evaluation output
        eval_output = {}
        eval_output['D'] = D
        eval_output['n'] = n
        eval_output['T'] = T
        
        return eval_output
        
    def log_eval(self, eval_output, **kwargs):
        # Create evaluation logger
        logger = Logger(name='eval_logger')
        
        # Unpack evaluation for logging
        D = eval_output['D']
        n = eval_output['n']
        T = eval_output['T']
        
        # Loggings: use item() to save memory
        # Log something about trajectories
        batch_returns = [sum(trajectory.all_r) for trajectory in D]
        batch_T = [trajectory.T for trajectory in D]
        
        logger.log('evaluation_iteration', n+1)
        logger.log('num_trajectories', len(D))
        logger.log('max_allowed_horizon', T)
        logger.log('average_horizon', np.mean(batch_T))
        logger.log('num_timesteps', np.sum(batch_T))
        logger.log('accumulated_trained_timesteps', self.agent.total_T)
        logger.log('average_return', np.mean(batch_returns))
        logger.log('std_return', np.std(batch_returns))
        logger.log('min_return', np.min(batch_returns))
        logger.log('max_return', np.max(batch_returns))
        
        # Dump loggings
        if n == 0 or (n+1) % self.config['log.print_interval'] == 0:
            print(color_str('+'*50, 'yellow', 'bold'))
            logger.dump(keys=None, index=None, indent=0)
            print(color_str('+'*50, 'yellow', 'bold'))

        return logger.logs
