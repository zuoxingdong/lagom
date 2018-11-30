from time import time
from itertools import chain

import numpy as np
import torch

from lagom import Logger
from lagom.utils import color_str

from lagom.envs.vec_env import get_wrapper
from lagom.envs.vec_env import VecStandardize
from lagom.runner import EpisodeRunner

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
        
        [logger(key, value) for key, value in out_agent.items()]
        
        logger('num_segments', D.N)
        logger('num_timesteps', D.total_T)
        logger('accumulated_trained_timesteps', self.agent.total_T)
        
        monitor_env = get_wrapper(self.runner.env, 'VecMonitor')
        infos = list(filter(lambda info: 'episode' in info, chain.from_iterable(D.infos)))
        if len(infos) > 0:
            online_returns = np.asarray([info['episode']['return'] for info in infos])
            online_horizons = np.asarray([info['episode']['horizon'] for info in infos])
            logger('online_N', len(infos))
            logger('online_mean_return', online_returns.mean())
            logger('online_std_return', online_returns.std())
            logger('online_min_return', online_returns.min())
            logger('online_max_return', online_returns.max())
            logger('online_mean_horizon', online_horizons.mean())
            logger('online_std_horizon', online_horizons.std())
            logger('online_min_horizon', online_horizons.min())
            logger('online_max_horizon', online_horizons.max())
        running_returns = np.asarray(monitor_env.return_queue)
        running_horizons = np.asarray(monitor_env.horizon_queue)
        if running_returns.size > 0 and running_horizons.size > 0:
            logger('running_queue', [len(monitor_env.return_queue), monitor_env.return_queue.maxlen])
            logger('running_mean_return', running_returns.mean())
            logger('running_std_return', running_returns.std())
            logger('running_min_return', running_returns.min())
            logger('running_max_return', running_returns.max())
            logger('running_mean_horizon', running_horizons.mean())
            logger('running_std_horizon', running_horizons.std())
            logger('running_min_horizon', running_horizons.min())
            logger('running_max_horizon', running_horizons.max())
        
        print('-'*50)
        logger.dump(keys=None, index=None, indent=0)
        print('-'*50)

        return logger.logs
        
    def eval(self, n):
        self.agent.eval()
        
        start_time = time()
        
        if self.config['env.standardize']:
            eval_env = VecStandardize(venv=self.eval_env,
                                      use_obs=True, 
                                      use_reward=False,  # do not process rewards, no training
                                      clip_obs=self.runner.env.clip_obs, 
                                      clip_reward=self.runner.env.clip_reward, 
                                      gamma=self.runner.env.gamma, 
                                      eps=self.runner.env.eps, 
                                      constant_obs_mean=self.runner.env.obs_runningavg.mu,
                                      constant_obs_std=self.runner.env.obs_runningavg.sigma)
        eval_runner = EpisodeRunner(self.config, self.agent, eval_env)
        T = eval_env.T
        D = eval_runner(T)
        
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
        logger('mean_horizon', D.Ts.mean())
        logger('total_timesteps', D.total_T)
        logger('accumulated_trained_timesteps', self.agent.total_T)
        logger('mean_return', batch_returns.mean())
        logger('std_return', batch_returns.std())
        logger('min_return', batch_returns.min())
        logger('max_return', batch_returns.max())
        
        print(color_str('+'*50, 'yellow', 'bold'))
        logger.dump(keys=None, index=None, indent=0)
        print(color_str('+'*50, 'yellow', 'bold'))

        return logger.logs
