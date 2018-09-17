from lagom.core.multiprocessing import BaseWorker


class BaseESWorker(BaseWorker):
    r"""Base class for the worker of parallelized evolution strategies (ES). 
    
    It defines an objective function to evaluate the given solution 
    candidate. 
    
    See :class:`BaseWorker` for more details about the workers.
    
    The subclass should implement at least the following:
    
    - :meth:`prepare`
    - :meth:`f`
    
    """
    def work(self, master_cmd):
        # Unpack master command
        solution_id, task, seed = master_cmd
        
        # Unpack task
        solution, config = task
        
        # Evaluate the solution to obtain fitness to the objective function
        function_value = self.f(solution, seed, config)
        
        return solution_id, function_value
    
    def f(self, solution, seed, config):
        r"""Defines an objective function to evaluate a given solution candidate. 
        
        .. note::
        
            In :attr:`solution`, it could contain additional information for the
            objective function evaluation. e.g. If it needs a gym environment for RL, 
            then :attr:`solution` should contain a function to create an environment. 
            e.g. ``solution, make_env = solution``
        
        Args:
            solution (object): given solution candidate. 
            seed (int): random seed contained in master_cmd. It can be used to seed 
                the current evaluation, e.g. gym environment. 
            config (dict): a dictionary of configurations
            
        Returns
        -------
        function_value : float
            objective function value for the given candidate. 
        """
        raise NotImplementedError
