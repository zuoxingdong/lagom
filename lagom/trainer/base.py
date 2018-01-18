class BaseTrainer(object):
    def __init__(self, agent, runner, args, logger=None):
        self.agent = agent
        self.runner = runner
        self.args = args
        self.logger = logger
        
    def train(self):
        raise NotImplementedError