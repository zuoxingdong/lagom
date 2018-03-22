from lagom.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """
    A random agent select action uniformly from action space
    """
    def __init__(self, env, config=None):
        self.env = env
        
        super().__init__(config)
        
    def choose_action(self, obs=None):
        # Dictionary of output data
        output = {}
        output['action'] = self.env.action_space.sample()
        
        return output
        
    def learn(self, batch_data=None):
        pass