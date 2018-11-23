from time import time
from collections import deque

import numpy as np

from .vec_env import VecEnvWrapper


class VecMonitor(VecEnvWrapper):
    r"""Record episode reward, horizon and time and report it when an episode terminates. """
    def __init__(self, venv, deque_size=100):
        super().__init__(venv)
        
        self.t0 = time()
        self.episode_rewards = np.zeros(self.N, dtype=np.float32)
        self.episode_horizons = np.zeros(self.N, dtype=np.int32)
        
        self.return_queue = deque(maxlen=deque_size)
        self.horizon_queue = deque(maxlen=deque_size)
        
    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        
        self.episode_rewards += rewards
        self.episode_horizons += 1
        for i, done in enumerate(dones):
            if done:
                infos[i]['episode'] = {'return': self.episode_rewards[i], 
                                       'horizon': self.episode_horizons[i],
                                       'time': round(time() - self.t0, 4)}
                self.return_queue.append(self.episode_rewards[i])
                self.horizon_queue.append(self.episode_horizons[i])
                
                self.episode_rewards[i] = 0.0
                self.episode_horizons[i] = 0
        
        return observations, rewards, dones, infos
        
    def reset(self):
        observations = self.venv.reset()
        
        self.episode_rewards.fill(0.0)
        self.episode_horizons.fill(0)
        
        return observations
        
    def close_extras(self):
        return self.venv.close_extras()
