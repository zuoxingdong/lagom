import numpy as np

from .env import Env
from .vec_env import VecEnv

from .spaces import Discrete
from .spaces import Box


class EnvSpec(object):
    """
    Create specifications of the environment. 
    
    Currently supported:
    - observation_space: observation space
    - action_space: action space
    - T: maximum allowed horizon
    - max_episode_reward: maximum episode reward
    - control_type: 'Discrete' or 'Continuous' control
    """
    def __init__(self, env):
        if not isinstance(env, (Env, VecEnv)):
            raise TypeError('The object env must be of type lagom Env or VecEnv.')
        
        self.env = env
    
    @property
    def T(self):
        return self.env.T
    
    @property
    def max_episode_reward(self):
        return self.env.max_episode_reward
        
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def control_type(self):
        if isinstance(self.env.action_space, Discrete):
            return 'Discrete'
        elif isinstance(self.env.action_space, Box):
            return 'Continuous'
        else:
            raise TypeError(f'expected type as Discrete or Box, got {type(self.env.action_space)}.')
