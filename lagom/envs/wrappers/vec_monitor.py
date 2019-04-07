from time import perf_counter
from collections import deque

import numpy as np

from lagom.envs import VecEnvWrapper


class VecMonitor(VecEnvWrapper):
    r"""Record episode reward, horizon and time and report it when an episode terminates. """
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        
        self.t0 = perf_counter()
        self.episode_rewards = np.zeros(len(env), dtype=np.float32)
        self.episode_horizons = np.zeros(len(env), dtype=np.int32)
        
        self.return_queue = deque(maxlen=deque_size)
        self.horizon_queue = deque(maxlen=deque_size)
        
    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)
        
        self.episode_rewards += rewards
        self.episode_horizons += 1
        for i, done in enumerate(dones):
            if done:
                infos[i]['episode'] = {'return': self.episode_rewards[i], 
                                       'horizon': self.episode_horizons[i],
                                       'time': round(perf_counter() - self.t0, 4)}
                self.return_queue.append(self.episode_rewards[i])
                self.horizon_queue.append(self.episode_horizons[i])
                
                self.episode_rewards[i] = 0.0
                self.episode_horizons[i] = 0
        
        return observations, rewards, dones, infos
        
    def reset(self):
        observations = self.env.reset()
        
        self.episode_rewards.fill(0.0)
        self.episode_horizons.fill(0)
        
        return observations
