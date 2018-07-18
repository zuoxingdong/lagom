import numpy as np

from .env import Env


class EnvSpec(object):
    """
    Create specifications of the environment
    """
    def __init__(self, env):
        if not isinstance(env, Env):
            raise TypeError('The object env must be of type lagom.envs.Env.')
        
        self.env = env
        
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
