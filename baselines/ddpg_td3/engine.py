import time

import torch
import lagom
import lagom.utils as utils


class Engine(lagom.BaseEngine):
    def train(self, n=None, **kwargs):
        self.agent.train()
        t0 = time.perf_counter()
        cond = utils.IntervalConditioner(interval=self.config['log.freq'], mode='accumulative')
        env = kwargs['env']
        runner = kwargs['runner']
        replay = kwargs['replay']
        
        if n < self.config['replay.init_trial']:
            [traj] = runner(lagom.RandomAgent(self.config, env), env, 1)
        else:
            [traj] = runner(self.agent, env, 1, mode='train')
        replay.add_trajectory(traj)
        # Number of gradient updates = collected episode length
        out_agent = self.agent.learn(D=None, replay=replay, T=traj.T)
        
        logger = lagom.Logger()
        logger('train_iteration', n+1)
        logger('num_seconds', round(time.perf_counter() - t0, 1))
        [logger(key, value) for key, value in out_agent.items()]
        logger('episode_return', sum(traj.rewards))
        logger('episode_horizon', traj.T)
        logger('accumulated_trained_timesteps', int(self.agent.total_timestep))
        
        if cond(n):
            logger.dump(keys=None, index=-1, indent=0, border='-'*50)
        return logger

    def eval(self, n=None, **kwargs):
        self.agent.eval()
        t0 = time.perf_counter()
        eval_env = kwargs['eval_env']
        runner = kwargs['runner']

        with torch.no_grad():
            D = runner(self.agent, eval_env, 10, mode='eval')
        
        logger = lagom.Logger()
        logger('eval_iteration', n+1)
        logger('num_seconds', round(time.perf_counter() - t0, 1))
        logger('accumulated_trained_timesteps', int(self.agent.total_timestep))
        logger('online_return', utils.describe([sum(traj.rewards) for traj in D], axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('online_horizon', utils.describe([traj.T for traj in D], axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('running_return', utils.describe(eval_env.return_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('running_horizon', utils.describe(eval_env.horizon_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger.dump(keys=None, index=-1, indent=0, border=utils.color_str('+'*50, color='green'))
        return logger
