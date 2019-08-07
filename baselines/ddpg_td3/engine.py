import time
from itertools import count

import torch
from lagom import Logger
from lagom import BaseEngine
from lagom.transform import describe
from lagom.utils import color_str


class Engine(BaseEngine):
    def train(self, n=None, **kwargs):
        train_logs, eval_logs = [], []
        checkpoint_count = 0
        for iteration in count():
            if self.agent.total_timestep >= self.config['train.timestep']:
                break
            t0 = time.perf_counter()
            
            if iteration < self.config['replay.init_trial']:
                [traj] = self.runner(self.random_agent, self.env, 1)
            else:
                [traj] = self.runner(self.agent, self.env, 1, mode='train')
            self.replay.add(traj)
            # Number of gradient updates = collected episode length
            out_agent = self.agent.learn(D=None, replay=self.replay, T=traj.T)
            
            logger = Logger()
            logger('train_iteration', iteration+1)
            logger('num_seconds', round(time.perf_counter() - t0, 1))
            [logger(key, value) for key, value in out_agent.items()]
            logger('episode_return', sum(traj.rewards))
            logger('episode_horizon', traj.T)
            logger('accumulated_trained_timesteps', self.agent.total_timestep)
            train_logs.append(logger.logs)
            if iteration == 0 or (iteration+1) % self.config['log.freq'] == 0:
                logger.dump(keys=None, index=0, indent=0, border='-'*50)
            if self.agent.total_timestep >= int(self.config['train.timestep']*(checkpoint_count/(self.config['checkpoint.num'] - 1))):
                self.agent.checkpoint(self.logdir, iteration + 1)
                checkpoint_count += 1
                
            if self.agent.total_timestep >= int(self.config['train.timestep']*(len(eval_logs)/(self.config['eval.num'] - 1))):
                eval_logs.append(self.eval(n=len(eval_logs)))
        return train_logs, eval_logs

    def eval(self, n=None, **kwargs):
        t0 = time.perf_counter()
        with torch.no_grad():
            D = self.runner(self.agent, self.eval_env, 10, mode='eval')
        
        logger = Logger()
        logger('eval_iteration', n+1)
        logger('num_seconds', round(time.perf_counter() - t0, 1))
        logger('accumulated_trained_timesteps', self.agent.total_timestep)
        logger('online_return', describe([sum(traj.rewards) for traj in D], axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('online_horizon', describe([traj.T for traj in D], axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('running_return', describe(self.eval_env.return_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('running_horizon', describe(self.eval_env.horizon_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger.dump(keys=None, index=0, indent=0, border=color_str('+'*50, color='green'))
        return logger.logs
