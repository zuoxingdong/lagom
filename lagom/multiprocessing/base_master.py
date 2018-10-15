from abc import ABC
from abc import abstractmethod


class BaseMaster(ABC):
    r"""Base class for all masters which assign tasks to a set of parallelized workers. 
    
    The subclass should implement at least the following:
    
    - :meth:`__call__`
    - :meth:`make_tasks`
    - :meth:`make_workers`
    - :meth:`assign_tasks`
    - :meth:`process_results`
    - :meth:`close`
    
    """ 
    @abstractmethod
    def __call__(self):
        r"""Defines the entire master-worker parallelization pipeline. """
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
        r"""Assign a given list of tasks to the created workers and return the received results. 
        
        Args:
            tasks (list): a list of tasks
            
        Returns
        -------
        results : object
            received results
        """
        pass
    
    @abstractmethod
    def process_results(self, results):
        r"""Process the results received from all the workers.
        
        Args:
            results (object): received results
        """
        pass
    
    @abstractmethod
    def close(self):
        r"""Defines everything required after finishing all the works, e.g. stop all workers, clean up. """
        pass
