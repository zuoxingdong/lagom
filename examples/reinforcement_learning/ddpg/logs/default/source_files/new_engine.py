from itertools import count

import numpy as np
import torch

from lagom import Logger
from lagom import BaseEngine
from lagom.envs.wrappers import get_wrapper
from lagom.utils import color_str


class Engine(BaseEngine):
    def train(self, n=None, **kwargs):
        self.agent.train()
        
        train_logs = []
        eval_logs = []
        eval_togo = 0
        num_episode = 0
        observation = self.env.reset()
        for i in count():
            if i >= self.config['train.timestep']:
                break
                
            if i < self.config['replay.init_size']:
                action = [self.env.action_space.sample()]
            else:
                
                action = self.agent.choose_action(self.replay.normalize_obs(observation))['action']
                
                
                
                
            next_observation, reward, done, info = self.env.step(action)
            eval_togo += 1
            if done[0]:  # [0] due to single environment
                if 'TimeLimit.truncated' in info[0]:  # NOTE: must use latest TimeLimit
                    reach_terminal = False
                else:
                    reach_terminal = done[0]
                self.replay.add(observation[0], action[0], reward[0], info[0]['last_observation'], reach_terminal)
                
                # DDPG updates in the end of episode, for each time step
                out_agent = self.agent.learn(D=None, replay=self.replay, episode_length=info[0]['episode']['horizon'])
                
                num_episode += 1
                logger = Logger()
                logger('accumulated_trained_timesteps', i + 1)
                logger('accumulated_trained_episodes', num_episode)
                [logger(key, value) for key, value in out_agent.items()]
                logger('episode_return', info[0]['episode']['return'])
                logger('episode_horizon', info[0]['episode']['horizon'])
                if num_episode % self.config['log.freq'] == 0:
                    logger.dump(keys=None, index=None, indent=0, border='-'*50)
                train_logs.append(logger.logs)
                
                if eval_togo >= self.config['eval.freq']:
                    eval_togo %= self.config['eval.freq']
                    eval_logs.append(self.eval(accumulated_trained_timesteps=(i+1), 
                                               accumulated_trained_episodes=num_episode))
            else:
                self.replay.add(observation[0], action[0], reward[0], next_observation[0], done[0])
            observation = next_observation
        return train_logs, eval_logs


    def eval(self, n=None, **kwargs):
        returns = []
        horizons = []
        for _ in range(self.config['eval.num_episode']):
            observation = self.eval_env.reset()
            for _ in range(self.eval_env.spec.max_episode_steps):
                observation = torch.from_numpy(np.asarray(observation)).float().to(self.agent.device)
                with torch.no_grad():
                    
                    
                    action = self.agent.actor(self.replay.normalize_obs(observation)).detach().cpu().numpy()
                    
                    
                    
                    
                next_observation, reward, done, info = self.eval_env.step(action)
                if done[0]:  # [0] single environment
                    returns.append(info[0]['episode']['return'])
                    horizons.append(info[0]['episode']['horizon'])
                    break
                observation = next_observation
        logger = Logger()
        logger('accumulated_trained_timesteps', kwargs['accumulated_trained_timesteps'])
        logger('accumulated_trained_episodes', kwargs['accumulated_trained_episodes'])
        logger('online_num_episode', len(returns))
        logger('online_mean_return', np.mean(returns))
        logger('online_std_return', np.std(returns))
        logger('online_min_return', np.min(returns))
        logger('online_max_return', np.max(returns))
        logger('online_mean_horizon', np.mean(horizons))
        logger('online_std_horizon', np.std(horizons))
        logger('online_min_horizon', np.min(horizons))
        logger('online_max_horizon', np.max(horizons))
        monitor_env = get_wrapper(self.eval_env, 'VecMonitor')
        running_returns = monitor_env.return_queue
        running_horizons = monitor_env.horizon_queue
        logger('running_queue', f'{len(running_returns)}/{running_returns.maxlen}')
        logger('running_mean_return', np.mean(running_returns))
        logger('running_std_return', np.std(running_returns))
        logger('running_min_return', np.min(running_returns))
        logger('running_max_return', np.max(running_returns))
        logger('running_mean_horizon', np.mean(running_horizons))
        logger('running_std_horizon', np.std(running_horizons))
        logger('running_min_horizon', np.min(running_horizons))
        logger('running_max_horizon', np.max(running_horizons))
        logger.dump(keys=None, index=None, indent=0, border=color_str('+'*50, color='green'))
        return logger.logs
