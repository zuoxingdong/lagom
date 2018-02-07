from lagom.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, env=None):
        self.env = env
        
    def choose_action(self, state):
        # Dictionary of output data
        output = {}
        output['action'] = self.env.action_space.sample()
        
        return output
        
    def learn(self, data_batch, standardize_r=False):
        pass