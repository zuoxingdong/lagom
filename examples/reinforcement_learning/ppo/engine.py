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
        
        D = self.runner(self.agent, self.env, self.config['train.timestep_per_iter'])
        out_agent = self.agent.learn(D) 
        
        logger = Logger()
        logger('train_iteration', n+1)
        logger('num_seconds', round(time() - start_time, 1))
        [logger(key, value) for key, value in out_agent.items()]
        logger('num_trajectories', len(D))
        logger('num_timesteps', sum([len(traj) for traj in D]))
        logger('accumulated_trained_timesteps', self.agent.total_timestep)
        G = [traj.numpy_rewards.sum() for traj in D]
        logger('mean_return', np.mean(G))
        logger('std_return', np.std(G))
        logger('min_return', np.min(G))
        logger('max_return', np.max(G))
        
        infos = chain.from_iterable([traj.infos for traj in D])
        online_returns = list(map(lambda x: x['episode']['return'], filter(lambda x: 'episode' in x, infos)))
        if len(online_returns) > 0:
            logger('online_mean_return', np.mean(online_returns))
            logger('online_std_return', np.std(online_returns))
            logger('online_min_return', np.min(online_returns))
            logger('online_max_return', np.max(online_returns))
        online_horizons = list(map(lambda x: x['episode']['horizon'], filter(lambda x: 'episode' in x, infos)))
        if len(online_horizons) > 0:
            logger('online_mean_horizon', np.mean(online_horizons))
            logger('online_std_horizon', np.std(online_horizons))
            logger('online_min_horizon', np.min(online_horizons))
            logger('online_max_horizion', np.max(online_horizons))
            
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
