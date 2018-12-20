from .base_agent import BaseAgent


class StickyAgent(BaseAgent):
    r"""An agent takes a sticky action regardless of observations. """
    def __init__(self, config, env_spec, sticky_action):
        super().__init__(config, env_spec, None)
        
        self.sticky_action = sticky_action
        
    def choose_action(self, obs, **kwargs):
        out = {}
        
        if self.env_spec.is_vec_env:
            action = [self.sticky_action]*self.env_spec.num_env
        else:
            action = self.sticky_action
        
        out['action'] = action
        
        return out
    
    def make_modules(self, config):
        pass
    
    def prepare(self, config, **kwargs):
        pass
    
    def reset(self, config, **kwargs):
        pass    
        
    def learn(self, D, **kwargs):
        pass
    
    @property
    def recurrent(self):
        pass
    
    def save(self, f):
        pass
    
    def load(self, f):
        pass
