from abc import ABC
from abc import abstractmethod

from lagom.multiprocessing import MPWorker


class BaseESWorker(MPWorker):
    r"""Base class for the worker of parallelized evolution strategies (ES). 
    
    It defines an objective function to evaluate the given solution 
    candidate. 
    
    The subclass should implement at least the following:
    
    - :meth:`prepare`
    - :meth:`f`
    
    """
    def work(self, task):
        task_id, task, use_chunk = task
        config, solution = task
        
        function_value = self.f(config, solution)
        
        return task_id, function_value
    
    @abstractmethod
    def f(self, config, solution):
        r"""Defines an objective function to evaluate a given solution candidate. 
        
        Args:
            config (dict): a dictionary of configurations
            solution (object): a given solution candidate. 
        
        Returns
        -------
        function_value : float
            objective function value for the given solution candidate. 
        """
        pass
