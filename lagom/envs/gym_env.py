import gym

from lagom.envs.base import Env

from lagom.envs.spaces import Box
from lagom.envs.spaces import Discrete
from lagom.envs.spaces import Product


class GymEnv(Env):
    def __init__(self, env):
        self.env = env
        
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        return self.env.reset()
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def clean(self):
        pass
    
    def get_source_env(self):
        return self.env
    
    @property
    def T(self):
        pass
    
    @property
    def observation_space(self):
        return self._convert_gym_space(self.env.observation_space)
    
    @property
    def action_space(self):
        return self._convert_gym_space(self.env.action_space)
    
    def _convert_gym_space(self, space):
        if isinstance(space, gym.spaces.Box):
            return Box(low=space.low, high=space.high)
        elif isinstance(space, gym.spaces.Discrete):
            return Discrete(n=space.n)
        elif isinstance(space, gym.spaces.Tuple):
            return Product([self._convert_gym_space(s) for s in space.spaces])
        else:
            raise NotImplementedError