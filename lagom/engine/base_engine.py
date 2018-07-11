class BaseEngine(object):
    """
    Base class for engine: agent training and evaluation process
    """
    def __init__(self, agent, runner, config, logger):
        self.agent = agent
        self.runner = runner
        self.config = config
        self.logger = logger
        
    def train(self):
        """
        Agent training process
        """
        raise NotImplementedError
        
    def eval(self):
        """
        Agent evaluation process
        """
        raise NotImplementedError