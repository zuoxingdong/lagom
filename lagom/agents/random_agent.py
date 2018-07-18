from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    A random agent samples action uniformly from action space
    """
    def __init__(self, env_spec, config=None, **kwargs):
        self.env_spec = env_spec
        
        super().__init__(config, **kwargs)
        
    def choose_action(self, obs):
        # Randomly sample an action from action space
        action = self.env_spec.action_space.sample()
        
        # Dictionary of output data
        output = {}
        output['action'] = action
        
        return output
        
    def learn(self, x):
        pass
    
    def save(self, filename):
        pass
    
    def load(self, filename):
        pass
