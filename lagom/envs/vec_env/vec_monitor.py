from time import time

import numpy as np

from .vec_env import VecEnvWrapper


class VecMonitor(VecEnvWrapper):
    r"""Record episode reward, length and time and report it when an episode terminates. """
    def __init__(self, venv):
        super().__init__(venv)
        
        self.t0 = time()
        self.episode_rewards = np.zeros(self.N, dtype=np.float32)
        self.episode_lengths = np.zeros(self.N, dtype=np.int32)
        
    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        
        self.episode_rewards += rewards
        self.episode_lengths += 1
        for i, done in enumerate(dones):
            if done:
                infos[i]['episode'] = {'return': self.episode_rewards[i], 
                                       'horizon': self.episode_lengths[i],
                                       'time': round(time() - self.t0, 4)}
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
        
        return observations, rewards, dones, infos
        
    def reset(self):
        observations = self.venv.reset()
        
        self.episode_rewards.fill(0.0)
        self.episode_lengths.fill(0)
        
        return observations
        
    def close_extras(self):
        return self.venv.close_extras()
