class BaseAgent(object):
    def __init__(self, policy, optimizer):
        self.policy = policy
        self.optimizer = optimizer
        
    def choose_action(self, state):
        raise NotImplementedError
        
    def learn(self, data_batch, standardize_r=False):
        raise NotImplementedError