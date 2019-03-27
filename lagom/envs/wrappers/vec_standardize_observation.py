import numpy as np

from lagom.transform import RunningMeanVar
from lagom.envs import VecEnvWrapper


class VecStandardizeObservation(VecEnvWrapper):
    r"""Standardizes the observations by running estimation of mean and variance. 
    
    .. warning::
    
        To evaluate the agent trained on standardized observations, remember to
        save and load observation scalings, otherwise, the performance will be incorrect. 
    
    Args:
        env (VecEnv): a vectorized environment
        clip (float): clipping range of standardized observation, i.e. [-clip, clip]
        constant_moments (tuple): a tuple of constant mean and variance to standardize observation.
            Note that if it is provided, then running average will be ignored.
    
    """
    def __init__(self, env, clip=10., constant_moments=None):
        super().__init__(env)
        self.clip = clip
        self.constant_moments = constant_moments
        
        self.eps = 1e-8
        
        if constant_moments is None:
            self.online = True
            self.running_moments = RunningMeanVar(shape=env.observation_space.shape)
        else:
            self.online = False
            self.constant_mean, self.constant_var = constant_moments    
        
    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)
        for i, info in enumerate(infos):  # standardize last_observation
            if 'last_observation' in info:
                infos[i]['last_observation'] = self.process_obs([info['last_observation']]).squeeze(0)
        return self.process_obs(observations), rewards, dones, infos
    
    def reset(self):
        observations = self.env.reset()
        return self.process_obs(observations)
    
    def process_obs(self, observations):
        if self.online:
            self.running_moments(observations)
            if self.running_moments.n >= 1:  # keep first observation unchanged due to zero std
                mean = self.running_moments.mean
                std = np.sqrt(self.running_moments.var + self.eps)
                observations = (observations - mean)/std
        else:
            mean = self.constant_mean
            std = np.sqrt(self.constant_var + self.eps)
            observations = (observations - mean)/std
        observations = np.clip(observations, -self.clip, self.clip)
        return observations.astype(np.float32)
    
    @property
    def mean(self):
        return self.running_moments.mean
    
    @property
    def var(self):
        return self.running_moments.var
