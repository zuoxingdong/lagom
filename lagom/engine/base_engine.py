from abc import ABC
from abc import abstractmethod


class BaseEngine(ABC):
    r"""Base class for all engines. 
    
    It defines the training and evaluation process. 
    
    The subclass should implement at least the following:
    
    - :meth:`train`
    - :meth:`log_train`
    - :meth:`eval`
    - :meth:`log_eval`
    
    """
    def __init__(self, agent, runner, config, **kwargs):
        self.agent = agent
        self.runner = runner
        self.config = config
        
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        
    @abstractmethod
    def train(self, n=None):
        r"""Agent training process for one iteration. 
        
        Args:
            n (int, optional): n-th iteration for training. 
        
        Returns
        -------
        train_output : dict
            training output
        """
        pass
        
    @abstractmethod
    def log_train(self, train_output, **kwargs):
        r"""Log the information about the training. 
        
        Internally, a Logger will be created and save all the logged information. 
        
        Args:
            train_output (dict): a dictionary of training output returned from :meth:`train`
            **kwargs: keyword aguments used for logging. 
            
        Returns
        -------
        train_logger : Logger
            logger with training information. 
        """
        pass
        
    @abstractmethod
    def eval(self, n=None):
        r"""Agent evaluation process for one iteration. 
        
        Args:   
            n (int, optional): n-th iteration for evaluation. 
        
        Returns
        -------
        eval_output : dict
            evluation output
        """
        pass
        
    @abstractmethod
    def log_eval(self, eval_output, **kwargs):
        r"""Log the information about evaluation. 
        
        Internally, a Logger will be created and save all the logged information. 
        
        Args:
            eval_output (dict): a dictionary of evaluation output returned from :meth:`eval`
            **kwargs: keyword aguments used for logging. 
            
        Returns
        -------
        eval_logger : Logger
            logger with evaluation information. 
        """
        pass
