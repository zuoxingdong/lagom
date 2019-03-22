

import numpy as np

from lagom import Logger
from lagom import BaseEngine
from lagom.utils import color_str


class Engine(BaseEngine):
    def train(self, n, **kwargs):
        self.agent.q_net.train()
        running_rewards = kwargs['running_rewards']
        step_counter = 0
        num_episode = 0
        episode_rewards = [0.0]
        
        while step_counter < self.config['train.timestep']:
            # run one full episode
            out_agent = self.agent.learn(D=None, replay=self.replay)
            observation = self.env.reset()
            while True:  # keep going until reaching a terminal state
                action = self.agent.choose_action(observation)['action']
                next_observation, reward, done, info = self.env.step(action)
                episode_rewards[-1] += reward[0]  # use [0] as single environment
                step_counter += 1

                if done[0]:
                    terminal_observation = info[0]['terminal_observation']
                    self.replay.add(observation[0], action[0], np.clip(reward[0], -1, 1), terminal_observation, done[0])
                else:
                    self.replay.add(observation[0], action[0], np.clip(reward[0], -1, 1), next_observation[0], done[0])
                observation = next_observation
                
                out_agent = self.agent.learn(D=None, replay=self.replay)

                if done[0]:
                    num_episode += 1
                    running_rewards.append(episode_rewards[-1])
                    episode_rewards.append(0.0)
                    break
        if episode_rewards[-1] == 0.0:
            episode_rewards = episode_rewards[:-1]
                    
        logger = Logger()
        logger('train_iteration', n + 1)
        logger('current_eps', self.agent.eps_scheduler.get_current())
        logger('accumulated_train_step', self.agent.train_step)
        if out_agent is not None:
            [logger(key, value) for key, value in out_agent.items()]
        logger('num_timestep', step_counter)
        logger('num_episode', num_episode)
        logger('mean_episode_return', np.mean(episode_rewards))
        logger('std_episode_return', np.std(episode_rewards))
        logger('min_episode_return', np.min(episode_rewards))
        logger('max_episode_return', np.max(episode_rewards))
        logger('running_mean_return', np.mean(running_rewards))
        logger('running_std_return', np.std(running_rewards))
        logger('running_min_return', np.min(running_rewards))
        logger('running_max_return', np.max(running_rewards))
        logger.dump(keys=None, index=None, indent=0, border='-'*50)
            
        return logger
    
    def eval(self, n=None, **kwargs):
        pass
