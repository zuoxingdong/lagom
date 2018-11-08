import numpy as np

import torch

from lagom import Logger
from lagom.utils import color_str

from lagom.engine import BaseEngine

from lagom.envs import make_gym_env
from lagom.envs import make_envs
from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.vec_env import VecStandardize


class Engine(BaseEngine):
    def train(self, n):
        self.agent.train()
        
        D = self.runner(T=self.config['train.T'])
        
        out_agent = self.agent.learn(D)
        
        train_output = {}
        train_output['D'] = D
        train_output['out_agent'] = out_agent
        train_output['n'] = n
        
        return train_output
        
    def log_train(self, train_output, **kwargs):
        D = train_output['D']
        out_agent = train_output['out_agent']
        n = train_output['n']
        
        logger = Logger()
        logger('train_iteration', n+1)  # starts from 1
        if 'current_lr' in out_agent:
            logger('current_lr', out_agent['current_lr'])
        logger('loss', out_agent['loss'])
        logger('policy_loss', out_agent['policy_loss'])
        logger('policy_entropy', -out_agent['entropy_loss'])
        logger('value_loss', out_agent['value_loss'])
        logger('explained_variance', out_agent['explained_variance'])
        
        batch_returns = [sum(trajectory.all_r) for trajectory in D]
        batch_discounted_returns = [trajectory.all_discounted_returns(self.config['algo.gamma'])[0] for trajectory in D]
        num_timesteps = sum([trajectory.T for trajectory in D])
        
        logger('num_trajectories', len(D))
        logger('num_timesteps', num_timesteps)
        logger('accumulated_trained_timesteps', self.agent.total_T)
        logger('average_return', np.mean(batch_returns))
        logger('average_discounted_return', np.mean(batch_discounted_returns))
        logger('std_return', np.std(batch_returns))
        logger('min_return', np.min(batch_returns))
        logger('max_return', np.max(batch_returns))

        if n == 0 or (n+1) % self.config['log.print_interval'] == 0:
            print('-'*50)
            logger.dump(keys=None, index=None, indent=0)
            print('-'*50)

        return logger.logs
        
    def eval(self, n):
        self.agent.eval()
        
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
        
        return eval_output
        
    def log_eval(self, eval_output, **kwargs):
        D = eval_output['D']
        n = eval_output['n']
        T = eval_output['T']
        
        logger = Logger()
        
        batch_returns = [sum(trajectory.all_r) for trajectory in D]
        batch_T = [trajectory.T for trajectory in D]
        
        logger('evaluation_iteration', n+1)
        logger('num_trajectories', len(D))
        logger('max_allowed_horizon', T)
        logger('average_horizon', np.mean(batch_T))
        logger('num_timesteps', np.sum(batch_T))
        logger('accumulated_trained_timesteps', self.agent.total_T)
        logger('average_return', np.mean(batch_returns))
        logger('std_return', np.std(batch_returns))
        logger('min_return', np.min(batch_returns))
        logger('max_return', np.max(batch_returns))
        
        if n == 0 or (n+1) % self.config['log.print_interval'] == 0:
            print(color_str('+'*50, 'yellow', 'bold'))
            logger.dump(keys=None, index=None, indent=0)
            print(color_str('+'*50, 'yellow', 'bold'))

        return logger.logs
