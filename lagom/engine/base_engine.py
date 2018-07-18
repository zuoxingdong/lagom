class BaseEngine(object):
    """
    Base class for engine: agent training and evaluation process. 
    
    All inherited subclasses should at least implement the following functions:
    1. train(self)
    2. eval(self)
    """
    def __init__(self, agent, runner, config, logger, **kwargs):
        self.agent = agent
        self.runner = runner
        self.config = config
        self.logger = logger
        
        # Set keyword arguments
        for key, val in kwargs.items():
            self.__setattr__(key, val)
        
    def train(self):
        """
        Agent training process
        
        Returns:
            train_output (dict): training output
        """
        raise NotImplementedError
        
    def eval(self):
        """
        Agent evaluation process
        
        Returns:
            eval_output (dict): evluation output
        """
        raise NotImplementedError
