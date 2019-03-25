from collections import deque

import numpy as np

import torch

from lagom import Logger
from lagom import BaseEngine
from lagom.envs.wrappers import get_wrapper
from lagom.envs.wrappers import VecStandardizeObservation
from lagom.utils import color_str


class Engine(BaseEngine):
    def train(self, n=None, **kwargs):
        self.agent.train()
        running_returns = deque(maxlen=100)
        step_counter = 0
        eval_togo = 0
        num_episode = 0
        train_logs = []
        eval_logs = []
        
        observation = self.env.reset()
        while step_counter < self.config['train.timestep']:
            if step_counter < self.config['replay.init_size']:
                action = [self.env.action_space.sample()]
            else:
                action = self.agent.choose_action(observation)['action']
            next_observation, reward, done, info = self.env.step(action)
            step_counter += 1
            eval_togo += 1
            
            if done[0]:  # [0] due to single environment
                terminal_observation = info[0]['terminal_observation']
                
                
                
                
                mean = get_wrapper(self.env, 'VecStandardizeObservation').mean
                std = get_wrapper(self.env, 'VecStandardizeObservation').std
                terminal_observation = np.clip((terminal_observation - mean)/(std + 1e-8), -10., 10.)
                
                
                
                episode_return = info[0]['episode']['return']
                episode_horizon = info[0]['episode']['horizon']
                if episode_horizon >= self.env.spec.max_episode_steps:
                    self.replay.add(observation[0], action[0], reward[0], terminal_observation, False)
                else:
                    self.replay.add(observation[0], action[0], reward[0], terminal_observation, done[0])
                
                # DDPG updates in the end of trajectory for each step
                out_agent = self.agent.learn(D=None, replay=self.replay, episode_length=episode_horizon)
                
                num_episode += 1
                running_returns.append(episode_return)
                
                logger = Logger()
                logger('accumulated_train_step', step_counter)
                [logger(key, value) for key, value in out_agent.items()]
                logger('total_num_episode', num_episode)
                logger('episode_return', episode_return)
                logger('episode_horizon', episode_horizon)
                logger('running_queue', f'{len(running_returns)}/{running_returns.maxlen}')
                logger('running_mean_return', np.mean(running_returns))
                logger('running_std_return', np.std(running_returns))
                logger('running_min_return', np.min(running_returns))
                logger('running_max_return', np.max(running_returns))
                if (num_episode + 1) % self.config['log.interval'] == 0:
                    logger.dump(keys=None, index=None, indent=0, border='-'*50)
                train_logs.append(logger.logs)
                
                if eval_togo >= self.config['eval.freq']:
                    eval_togo %= self.config['eval.freq']
                    eval_logs.append(self.eval(accumulated_train_step=step_counter))
            else:
                self.replay.add(observation[0], action[0], reward[0], next_observation[0], done[0])
            observation = next_observation
        return train_logs, eval_logs
    
    def eval(self, n=None, **kwargs):
        returns = []
        horizons = []
        if self.config['env.standardize']:
            mean = get_wrapper(self.env, 'VecStandardizeObservation').mean
            std = get_wrapper(self.env, 'VecStandardizeObservation').std
            eval_env = VecStandardizeObservation(self.eval_env, clip=10., constant_mean=mean, constant_std=std)
        else:
            eval_env = self.eval_env
        for _ in range(self.config['eval.num_episode']):
            observation = eval_env.reset()
            for _ in range(eval_env.spec.max_episode_steps):
                observation = torch.from_numpy(np.asarray(observation)).float().to(self.agent.device)
                with torch.no_grad():
                    action = self.agent.actor(observation).detach().cpu().numpy()
                next_observation, reward, done, info = eval_env.step(action)
                if done[0]:  # [0] single environment
                    returns.append(info[0]['episode']['return'])
                    horizons.append(info[0]['episode']['horizon'])
                    break
                observation = next_observation
        logger = Logger()
        logger('accumulated_train_step', kwargs['accumulated_train_step'])
        logger('eval_num_episode', len(returns))
        logger('eval_mean_return', np.mean(returns))
        logger('eval_std_return', np.std(returns))
        logger('eval_min_return', np.min(returns))
        logger('eval_max_return', np.max(returns))
        logger('eval_mean_horizon', np.mean(horizons))
        logger('eval_std_horizon', np.std(horizons))
        logger('eval_min_horizon', np.min(horizons))
        logger('eval_max_horizon', np.max(horizons))
        logger.dump(keys=None, index=None, indent=0, border=color_str('+'*50, color='green'))
        return logger.logs