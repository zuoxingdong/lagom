from lagom.core.multiprocessing import BaseWorker


class BaseESWorker(BaseWorker):
    """
    Base class for the worker of parallelized evolution strategies (ES). 
    
    It defines an objective function to evaluate the given solution 
    candidate and compute a objective function value. 
    
    For more details about how worker class works, please refer
    to the documentation of the class, BaseWorker. 
    
    All inherited subclasses should implement the following function:
    1. f(self, solution)
    """
    def work(self, master_cmd):
        # Unpack master command
        solution_id, solution, seed = master_cmd
        
        # Evaluate the solution to obtain fitness to the objective function
        function_value = self.f(solution, seed)
        
        return solution_id, function_value
    
    def f(self, solution, seed):
        """
        User-defined function to define the objective function given
        the solution candidate. 
        
        Note that the solution argument can contain additional information 
        needed for evaluating the objective function value. For example, 
        if the usecase is doing RL with gym environments, then solution
        could contain a function to create an environment. 
        e.g. `solution, make_env = solution`
        
        Args:
            solution (object): given solution candidate. 
            seed (int): random seed contained in master_cmd. 
                It can be used to seed the current evaluation, e.g. gym environment. 
            
        Returns:
            function_value (float): objective function value
        """
        raise NotImplementedError