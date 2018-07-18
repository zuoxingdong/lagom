import gym

from .env import Env

from spaces import Box
from spaces import Dict
from spaces import Discrete
from spaces import Product


class GymEnv(Env):
    def __init__(self, env):
        self.env = env
        
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        return self.env.reset()
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def close(self):
        return self.env.close()
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    @property
    def unwrapped(self):
        return self.env.unwrapped
    
    @property
    def T(self):
        return self.env.T
    
    @property
    def observation_space(self):
        return self._convert_gym_space(self.env.observation_space)
    
    @property
    def action_space(self):
        return self._convert_gym_space(self.env.action_space)
    
    def _convert_gym_space(self, space):
        """
        Enforce the space dtype to lagom spaces
        
        Args:
            space (gym Space): gym version of spaces
            
        Returns:
            converted space (lagom Space)
        """
        if isinstance(space, gym.spaces.Box):
            return Box(low=space.low, high=space.high, dtype=space.dtype)  # Don't give shape
        elif isinstance(space, gym.spaces.Dict):
            return Dict(dict([(key, self._convert_gym_space(space)) for key, space in space.spaces.items()]))
        elif isinstance(space, gym.spaces.Discrete):
            return Discrete(n=space.n)
        elif isinstance(space, gym.spaces.Tuple):
            return Product([self._convert_gym_space(s) for s in space.spaces])
        else:
            raise TypeError('Currently only Box, Dict, Discrete and Tuple spaces are supported. ')
