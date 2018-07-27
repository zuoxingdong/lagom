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
        
    def train(self, n=None):
        """
        Agent training process for one iteration. 
        
        Args:
            n (int, optional): n-th iteration for training. 
        
        Returns:
            train_output (dict): training output
        """
        raise NotImplementedError
        
    def log_train(self, train_output, **kwargs):
        """
        Log the information about the training. 
        
        Internally, a Logger will be created and save all the logged information. 
        
        Args:
            train_output (dict): dictionary of training output returned from `train()`
            **kwargs: keyword aguments used for logging. 
            
        Returns:
            train_logger (Logger): logger with training information. 
        """
        raise NotImplementedError
        
    def eval(self, n=None):
        """
        Agent evaluation process for one iteration. 
        
        Args:   
            n (int, optional): n-th iteration for evaluation. 
        
        Returns:
            eval_output (dict): evluation output
        """
        raise NotImplementedError
        
    def log_eval(self, eval_output, **kwargs):
        """
        Log the information about evaluation. 
        
        Internally, a Logger will be created and save all the logged information. 
        
        Args:
            eval_output (dict): dictionary of evaluation output returned from `eval()`
            **kwargs: keyword aguments used for logging. 
            
        Returns:
            eval_logger (Logger): logger with evaluation information. 
        """
        raise NotImplementedError
