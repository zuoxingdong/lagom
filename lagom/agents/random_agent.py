from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    r"""A random agent samples action uniformly from action space. 
    
    Example::
    
    
    """
    def __init__(self, config, env_spec):
        super().__init__(config, env_spec, None)
        
    def choose_action(self, obs, info={}):
        out = {}
        
        if self.env_spec.is_vec_env:
            action = [self.action_space.sample() for _ in range(self.env_spec.env.num_env)]
        else:
            action = self.action_space.sample()
        
        out['action'] = action
        
        return out
    
    def make_modules(self, config):
        pass
    
    def prepare(self, config, **kwargs):
        pass
    
    def reset(self, config, **kwargs):
        pass    
        
    def learn(self, D, info={}):
        pass
    
    @property
    def recurrent(self):
        pass
    
    def save(self, f):
        pass
    
    def load(self, f):
        pass
