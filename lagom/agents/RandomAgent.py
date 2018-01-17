from lagom.agents.BaseAgent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self):
        pass
        
    def choose_action(self, state):
        env = state.get('env', None)
        
        if env is None:
            raise TypeError('The input must contain the Gym environment object to support random action.')
        
        # Dictionary of output data
        output = {}
        output['action'] = env.action_space.sample()
        
        return output
        
    def train(self, data_batch, normalize_r=False):
        pass