import numpy as np

from lagom.transform import RunningMeanStd
from lagom.envs import VecEnvWrapper


class VecStandardizeObservation(VecEnvWrapper):
    r"""Standardizes the observations by running averages. 
    
    Each observation is substracted from the running mean and devided by running standard deviation. 
    
    .. warning::
    
        To evaluate the agent trained on standardized observations, remember to
        save and load observation scalings, otherwise, the performance will be incorrect. 
    
    Args:
        env (VecEnv): a vectorized environment
        clip (float): clipping range of standardized observation, i.e. [-clip, clip]
        constant_mean (ndarray): Constant mean to standardize observation. Note that
            when it is provided, then running average will be ignored.
        constant_std (ndarray): Constant standard deviation to standardize observation. Note that
            when it is provided, then running average will be ignored. 
    
    """
    def __init__(self, env, clip=10., constant_mean=None, constant_std=None):
        super().__init__(env)
        self.clip = clip
        self.constant_mean = constant_mean
        self.constant_std = constant_std
        
        self.eps = 1e-8
        self.runningavg = RunningMeanStd()
        
    def step_wait(self):
        observations, rewards, dones, infos = self.env.step_wait()
        return self.process_obs(observations), rewards, dones, infos
    
    def reset(self):
        observations = self.env.reset()
        return self.process_obs(observations)
    
    def process_obs(self, observations):
        obs = np.asarray(observations)
        if self.constant_mean is None and self.constant_std is None:
            self.runningavg(obs)
            mean = np.expand_dims(self.runningavg.mu, 0)  # add batch dim for safe broadcast
            std = np.expand_dims(self.runningavg.sigma, 0)
        elif self.constant_mean is not None and self.constant_std is not None:
            mean = np.expand_dims(self.constant_mean, 0)  # add batch dim for safe broadcast
            std = np.expand_dims(self.constant_std, 0)
        else:
            raise ValueError
        
        obs = (obs - mean)/(std + self.eps)
        obs = np.clip(obs, -self.clip, self.clip)
        
        return obs
    
    @property
    def mean(self):
        return self.runningavg.mu
    
    @property
    def std(self):
        return self.runningavg.sigma
