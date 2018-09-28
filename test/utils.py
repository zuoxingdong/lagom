import numpy as np

from lagom.envs import Env
from lagom.envs.spaces import Box
from lagom.envs.spaces import Discrete

from functools import partial


class SanityCheckEnv(Env):
    r"""Environment for a robust sanity check of some implementations. 
    
    Example::
    
        >>> env = SanityCheckEnv(2)
        >>> env.reset()
        0.0
        
        >>> env.step(env.action_space.sample())
        (1.0, 0.1, False, {})
        
        >>> env.step(env.action_space.sample())
        (2.0, 0.2, True, {})
    
    """
    def __init__(self, cycle):
        self.cycle = cycle
        
        self.x = 0.01
        self.r = 0.0
        self.count = 0
        
        self._observation_space = Box(0.0, float(self.cycle), shape=[1,], dtype=np.float32)
        self._action_space = Discrete(2)
        
    def step(self, action):
        self.x += 1.0
        observation = self.x
        
        self.r += 0.1
        reward = self.r
        
        self.count += 1
        if self.count >= self.cycle:
            done = True
        else:
            done = False
        
        info = {}
        
        return [observation], reward, done, info
        
    def reset(self):
        self.x = 0.01
        self.r = 0.0
        self.count = 0
        
        return [self.x]

    def render(self, mode='human'):
        pass
        
    def close(self):
        pass

    def seed(self, seed):
        pass

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def T(self):
        return self.cycle

    @property
    def max_episode_reward(self):
        return np.sum(np.arange(1, self.T)/10)

    @property
    def reward_range(self):
        return [0.1, self.cycyle/10]


def make_sanity_env(T, env_id):
    return SanityCheckEnv(T)


def make_sanity_envs(list_T):
    return [partial(make_sanity_env, T=T, env_id='SanityCheck-v0') for T in list_T]
