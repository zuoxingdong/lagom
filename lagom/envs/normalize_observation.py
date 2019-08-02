import numpy as np
import gym

from lagom.transform import RunningMeanVar


class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, clip=5., constant_moments=None):
        super().__init__(env)
        self.clip = clip
        self.constant_moments = constant_moments
        self.eps = 1e-8
        if constant_moments is None:
            self.obs_moments = RunningMeanVar(shape=env.observation_space.shape)
        else:
            self.constant_mean, self.constant_var = constant_moments
            
    def observation(self, observation):
        if self.constant_moments is None:
            self.obs_moments([observation])
            mean = self.obs_moments.mean
            std = np.sqrt(self.obs_moments.var + self.eps)
        else:
            mean = self.constant_mean
            std = np.sqrt(self.constant_var + self.eps)
        observation = np.clip((observation - mean)/std, -self.clip, self.clip)
        return observation
