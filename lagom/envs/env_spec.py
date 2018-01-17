import numpy as np


class EnvSpec(object):
    """
    Create specifications of the environment
    """
    def __init__(self, env):
        self.env = env
        
        self.env_spec = {}
        self.env_spec['obs_dim'] = int(np.prod(self.env.observation_space.shape))
        self.env_spec['state_dim'] = None
        self.env_spec['action_dim'] = self.env.action_space.n
        
        if hasattr(env, 'goal_states'):
            self.env_spec['goal_dim'] = np.array(env.goal_states).reshape(-1).shape[0]
        
    def get(self, key):
        return self.env_spec.get(key, None)
    
    def set(self, key, value):
        self.env_spec[key] = value
