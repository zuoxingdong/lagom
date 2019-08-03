from abc import ABC
from abc import abstractmethod


class BaseEngine(ABC):
    r"""Base class for all engines. 
    
    It defines the training and evaluation process. 
    
    """
    def __init__(self, config, **kwargs):
        self.config = config
        
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        
    @abstractmethod
    def train(self, n=None, **kwargs):
        r"""Training process for one iteration. 
        
        .. note::
        
            It is recommended to use :class:`Logger` to store loggings. 
            
        .. note::
        
            All parameterized modules should be called `.train()` to specify training mode.
        
        Args:
            n (int, optional): n-th iteration for training. 
            **kwargs: keyword aguments used for logging. 
        
        Returns:
            dict: a dictionary of training output    
        """
        pass
        
    @abstractmethod
    def eval(self, n=None, **kwargs):
        r"""Evaluation process for one iteration. 
        
        .. note::
        
            It is recommended to use :class:`Logger` to store loggings. 
            
        .. note::
        
            All parameterized modules should be called `.eval()` to specify evaluation mode.
        
        Args:   
            n (int, optional): n-th iteration for evaluation. 
            **kwargs: keyword aguments used for logging. 
        
        Returns:
            dict: a dictionary of evluation output
        """
        pass
