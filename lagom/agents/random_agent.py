from .base_agent import BaseAgent

from lagom.envs.vec_env import VecEnv


class RandomAgent(BaseAgent):
    """
    A random agent samples action uniformly from action space. 
    
    Note that the agent should handle batched data. 
    """
    def __init__(self, env_spec, config=None, **kwargs):
        self.env_spec = env_spec
        
        super().__init__(config, **kwargs)
        
    def choose_action(self, obs):
        # Randomly sample an action from action space
        if isinstance(self.env_spec.env, VecEnv):
            num_env = self.env_spec.env.num_env
            action = [self.env_spec.action_space.sample() for _ in range(num_env)]
        else:
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
