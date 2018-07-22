import numpy as np

from .env import Env
from .vec_env import VecEnv


class EnvSpec(object):
    """
    Create specifications of the environment
    """
    def __init__(self, env):
        if not isinstance(env, (Env, VecEnv)):
            raise TypeError('The object env must be of type lagom Env or VecEnv.')
        
        self.env = env
        
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
