from time import perf_counter
from itertools import count

import numpy as np
import torch

from lagom import Logger
from lagom import BaseEngine
from lagom.transform import describe
from lagom.utils import color_str
from lagom.envs.wrappers import get_wrapper


class Engine(BaseEngine):
    def train(self, n=None, **kwargs):
        train_logs = []
        eval_logs = []
        eval_togo = 0
        checkpoint_togo = 0
        num_episode = 0
        observation = self.env.reset()
        for i in count():
            if i >= self.config['train.timestep']:
                break
                
            if i < self.config['replay.init_size']:
                action = [self.env.action_space.sample()]
            else:
                action = self.agent.choose_action(observation, mode='train')['action']
            next_observation, reward, done, info = self.env.step(action)
            eval_togo += 1
            checkpoint_togo += 1
            if done[0]:  # [0] due to single environment
                start_time = perf_counter()
                # NOTE: must use latest TimeLimit
                reach_time_limit = info[0].get('TimeLimit.truncated', False)
                reach_terminal = not reach_time_limit
                self.replay.add(observation[0], action[0], reward[0], info[0]['last_observation'], reach_terminal)
                
                # updates in the end of episode, for each time step
                out_agent = self.agent.learn(D=None, replay=self.replay, episode_length=info[0]['episode']['horizon'])
                num_episode += 1
                if checkpoint_togo >= self.config['checkpoint.freq']:
                    checkpoint_togo %= self.config['checkpoint.freq']
                    self.agent.checkpoint(self.logdir, num_episode)
                
                logger = Logger()
                logger('num_seconds', round(perf_counter() - start_time, 1))
                logger('accumulated_trained_timesteps', i + 1)
                logger('accumulated_trained_episodes', num_episode)
                [logger(key, value) for key, value in out_agent.items()]
                logger('episode_return', info[0]['episode']['return'])
                logger('episode_horizon', info[0]['episode']['horizon'])
                train_logs.append(logger.logs)
                if num_episode == 1 or num_episode % self.config['log.freq'] == 0:
                    logger.dump(keys=None, index=0, indent=0, border='-'*50)
                
                if eval_togo >= self.config['eval.freq']:
                    eval_togo %= self.config['eval.freq']
                    eval_logs.append(self.eval(accumulated_trained_timesteps=(i+1), 
                                               accumulated_trained_episodes=num_episode))
            else:
                self.replay.add(observation[0], action[0], reward[0], next_observation[0], done[0])
            observation = next_observation
        return train_logs, eval_logs

    def eval(self, n=None, **kwargs):
        start_time = perf_counter()
        returns = []
        horizons = []
        for _ in range(self.config['eval.num_episode']):
            observation = self.eval_env.reset()
            for _ in range(self.eval_env.spec.max_episode_steps):
                with torch.no_grad():
                    action = self.agent.choose_action(observation, mode='eval')['action']
                next_observation, reward, done, info = self.eval_env.step(action)
                if done[0]:  # [0] single environment
                    returns.append(info[0]['episode']['return'])
                    horizons.append(info[0]['episode']['horizon'])
                    break
                observation = next_observation
        logger = Logger()
        logger('num_seconds', round(perf_counter() - start_time, 1))
        logger('accumulated_trained_timesteps', kwargs['accumulated_trained_timesteps'])
        logger('accumulated_trained_episodes', kwargs['accumulated_trained_episodes'])
        logger('online_return', describe(returns, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('online_horizon', describe(horizons, axis=-1, repr_indent=1, repr_prefix='\n'))
        
        monitor_env = get_wrapper(self.eval_env, 'VecMonitor')
        logger('running_return', describe(monitor_env.return_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('running_horizon', describe(monitor_env.horizon_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger.dump(keys=None, index=0, indent=0, border=color_str('+'*50, color='green'))
        return logger.logs
