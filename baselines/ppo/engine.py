import time

from lagom import Logger
from lagom import BaseEngine
from lagom.transform import describe
from lagom.utils import IntervalConditioner


class Engine(BaseEngine):        
    def train(self, n=None, **kwargs):
        self.agent.train()
        t0 = time.perf_counter()
        cond = IntervalConditioner(interval=self.config['log.freq'], mode='accumulative')
        
        D = self.runner(self.agent, self.env, self.config['train.timestep_per_iter'])
        out_agent = self.agent.learn(D) 
        
        logger = Logger()
        logger('train_iteration', n+1)
        logger('num_seconds', round(time.perf_counter() - t0, 1))
        [logger(key, value) for key, value in out_agent.items()]
        logger('num_trajectories', len(D))
        logger('num_timesteps', sum([traj.T for traj in D]))
        logger('accumulated_trained_timesteps', int(self.agent.total_timestep))
        logger('return', describe([sum(traj.rewards) for traj in D], axis=-1, repr_indent=1, repr_prefix='\n'))
        
        E = [traj[-1].info['episode'] for traj in D if 'episode' in traj[-1].info]
        logger('online_return', describe([e['return'] for e in E], axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('online_horizon', describe([e['horizon'] for e in E], axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('running_return', describe(self.env.return_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('running_horizon', describe(self.env.horizon_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        
        if cond(n):
            logger.dump(keys=None, index=-1, indent=0, border='-'*50)
        return logger
        
    def eval(self, n=None, **kwargs):
        pass
