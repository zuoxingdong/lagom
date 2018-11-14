from time import time

import numpy as np
import torch

from lagom import Logger
from lagom.utils import color_str

from lagom.engine import BaseEngine


class Engine(BaseEngine):
    def train(self, n):
        self.agent.train()
        
        start_time = time()
        
        T = int(self.config['train.ratio_T']*self.runner.env.T)
        D = self.runner(T)
        
        out_agent = self.agent.learn(D)
        
        train_output = {}
        train_output['D'] = D
        train_output['out_agent'] = out_agent
        train_output['n'] = n
        train_output['num_sec'] = time() - start_time
        
        return train_output

    def log_train(self, train_output, **kwargs):
        D = train_output['D']
        out_agent = train_output['out_agent']
        n = train_output['n']
        num_sec = train_output['num_sec']
        
        logger = Logger()
        logger('train_iteration', n+1)  # starts from 1
        logger('num_seconds', round(num_sec, 1))
        
        if 'current_lr' in out_agent:
            logger('current_lr', out_agent['current_lr'])
        logger('loss', out_agent['loss'])
        logger('policy_loss', out_agent['policy_loss'])
        logger('policy_entropy', -out_agent['entropy_loss'])
        logger('value_loss', out_agent['value_loss'])
        logger('explained_variance', out_agent['explained_variance'])
        
        batch_returns = D.numpy_rewards.sum(1)
        
        logger('num_trajectories', D.N)
        logger('num_timesteps', D.total_T)
        logger('accumulated_trained_timesteps', self.agent.total_T)
        logger('average_return', batch_returns.mean())
        logger('std_return', batch_returns.std())
        logger('min_return', batch_returns.min())
        logger('max_return', batch_returns.max())
        
        if n == 0 or (n+1) % self.config['log.print_interval'] == 0:
            print('-'*50)
            logger.dump(keys=None, index=None, indent=0)
            print('-'*50)

        return logger.logs
        
    def eval(self, n):
        self.agent.eval()
        
        start_time = time()
        
        # Synchronize running average of observations for evaluation
        if self.config['env.standardize']:
            self.eval_runner.env.constant_obs_mean = self.runner.env.obs_runningavg.mu
            self.eval_runner.env.constant_obs_std = self.runner.env.obs_runningavg.sigma
        
        T = self.eval_runner.env.T
        D = self.eval_runner(T)
        
        eval_output = {}
        eval_output['D'] = D
        eval_output['n'] = n
        eval_output['T'] = T
        eval_output['num_sec'] = time() - start_time
        
        return eval_output
        
    def log_eval(self, eval_output, **kwargs):
        D = eval_output['D']
        n = eval_output['n']
        T = eval_output['T']
        num_sec = eval_output['num_sec']
        
        logger = Logger()
        
        batch_returns = D.numpy_rewards.sum(1)
        
        logger('evaluation_iteration', n+1)
        logger('num_seconds', round(num_sec, 1))
        logger('num_trajectories', D.N)
        logger('max_allowed_horizon', T)
        logger('average_horizon', D.Ts.mean())
        logger('total_timesteps', D.total_T)
        logger('accumulated_trained_timesteps', self.agent.total_T)
        logger('average_return', batch_returns.mean())
        logger('std_return', batch_returns.std())
        logger('min_return', batch_returns.min())
        logger('max_return', batch_returns.max())
        
        if n == 0 or (n+1) % self.config['log.print_interval'] == 0:
            print(color_str('+'*50, 'yellow', 'bold'))
            logger.dump(keys=None, index=None, indent=0)
            print(color_str('+'*50, 'yellow', 'bold'))

        return logger.logs
