from .base_agent import BaseAgent

from lagom.envs.vec_env import VecEnv


class RandomAgent(BaseAgent):
    r"""A random agent samples action uniformly from action space. """
    def __init__(self, config, env_spec, **kwargs):
        self.env_spec = env_spec
        
        super().__init__(config=config, device=None, **kwargs)
        
    def choose_action(self, obs, info={}):
        out = {}
        
        # Randomly sample an action from action space
        if isinstance(self.env_spec.env, VecEnv):
            num_env = self.env_spec.env.num_env
            action = [self.env_spec.action_space.sample() for _ in range(num_env)]
        else:
            action = self.env_spec.action_space.sample()
        out['action'] = action
        
        return out
        
    def learn(self, D, info={}):
        pass
    
    def save(self, f):
        pass
    
    def load(self, f):
        pass
    
    def __repr__(self):
        string = super().__repr__()
        string += f'\n\tEnvironment specification: {self.env_spec}'
        
        return string
