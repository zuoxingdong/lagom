class Agent(object):
    def __init__(self, policy, optimizer):
        self.policy = policy
        self.optimizer = optimizer
        
    def choose_action(self, state):
        raise NotImplementedError
        
    def train(self, data_batch, normalize_r=False):
        raise NotImplementedError