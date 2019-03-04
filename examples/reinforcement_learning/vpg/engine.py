from time import time
from itertools import chain

import numpy as np

from lagom import Logger
from lagom import BaseEngine
from lagom.utils import color_str
from lagom.envs.wrappers import get_wrapper


class Engine(BaseEngine):        
    def train(self, n=None, **kwargs):
        self.agent.train()
        start_time = time()
        
        T = int(self.config['train.ratio_T']*self.env.spec.max_episode_steps)
        D = self.runner(self.agent, self.env, T)
        out_agent = self.agent.learn(D) 
        
        logger = Logger()
        logger('train_iteration', n+1)
        logger('num_seconds', round(time() - start_time, 1))
        [logger(key, value) for key, value in out_agent.items()]
        logger('num_trajectories', sum(D.num_traj))
        logger('num_timesteps', sum(D.Ts_flat))
        logger('accumulated_trained_timesteps', self.agent.total_T)
        G = D.rewards.sum(1)
        logger('mean_return', G.mean())
        logger('std_return', G.std())
        logger('min_return', G.min())
        logger('max_return', G.max())
        
        infos = list(filter(lambda x: 'episode' in x, 
                            chain.from_iterable([chain.from_iterable(info) for info in D.info])))
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
        monitor_env = get_wrapper(self.env, 'VecMonitor')
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
        
        return logger
        
    def eval(self, n=None, **kwargs):
        pass
