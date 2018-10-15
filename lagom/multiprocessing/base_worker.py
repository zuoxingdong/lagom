from abc import ABC
from abc import abstractmethod


class BaseWorker(ABC):
    r"""Base class for all workers who receive tasks from master and send back the result. 
    
    The subclass should implement at least the following:
    
    - :meth:`__call__`
    - :meth:`prepare`
    - :meth:`work`
    
    """
    @abstractmethod
    def __call__(self):
        r"""Defines the pipeline for the working. """
        pass
    
    @abstractmethod
    def prepare(self):
        r"""Defines the preparations for the work. 
        
        For example, an environment object can be created only once and each successive working received
        from master can use the same environment. This might be much more efficient than creating a new
        one every time. 
        """
        pass
        
    @abstractmethod
    def work(self, task):
        r"""Work on the given task and return the result. 
        
        Args:
            task (object): a given task. 
            
        Returns
        -------
        result : object
            working result. 
        """
        pass
