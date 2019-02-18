from abc import ABC
from abc import abstractmethod


class BaseMaster(ABC):
    r"""Base class for the master in master-worker architecture which assigns tasks 
    to a set of parallelized workers. 
    """ 
    @abstractmethod
    def __call__(self):
        r"""Defines the master-worker parallelization pipeline. """
        pass
        
    @abstractmethod
    def make_tasks(self):
        r"""Returns a list of tasks. 
        
        Returns
        -------
        tasks : list
            a list of tasks
        """
        pass
    
    @abstractmethod
    def make_workers(self):
        r"""Create multiple workers. """
        pass
    
    @abstractmethod
    def assign_tasks(self, tasks):
        r"""Assign a given list of tasks to the workers and return the received results. 
        
        Args:
            tasks (list): a list of tasks
            
        Returns
        -------
        results : object
            received results
        """
        pass
    
    @abstractmethod
    def close(self):
        r"""Defines everything required after finishing all the works, e.g. stop all workers, clean up. """
        pass
