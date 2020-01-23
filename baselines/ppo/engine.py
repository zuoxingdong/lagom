import time

import lagom
import lagom.utils as utils


class Engine(lagom.BaseEngine):        
    def train(self, n=None, **kwargs):
        self.agent.train()
        t0 = time.perf_counter()
        cond_log = kwargs['cond_log']
        
        D = self.runner(self.agent, self.env, self.config['train.timestep_per_iter'])
        out_agent = self.agent.learn(D) 
        
        logger = lagom.Logger()
        logger('train_iteration', n+1)
        logger('num_seconds', round(time.perf_counter() - t0, 1))
        [logger(key, value) for key, value in out_agent.items()]
        logger('num_trajectories', len(D))
        logger('num_timesteps', sum([traj.T for traj in D]))
        logger('accumulated_trained_timesteps', int(self.agent.total_timestep))
        describe_it = lambda x: utils.describe(x, axis=-1, repr_indent=1, repr_prefix='\n')
        logger('return', describe_it([sum(traj.rewards) for traj in D]))
        E = [traj[-1].info['episode'] for traj in D if 'episode' in traj[-1].info]
        logger('online_return', describe_it([e['return'] for e in E]))
        logger('online_horizon', describe_it([e['horizon'] for e in E]))
        logger('running_return', describe_it(self.env.return_queue))
        logger('running_horizon', describe_it(self.env.horizon_queue))
        if cond_log(n):
            logger.dump(keys=None, index=-1, indent=0, border='-'*50)
        return logger
        
    def eval(self, n=None, **kwargs):
        pass
