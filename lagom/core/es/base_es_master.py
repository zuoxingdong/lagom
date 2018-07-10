from lagom.core.multiprocessing import BaseIterativeMaster


class BaseESMaster(BaseIterativeMaster):
    """
    Base class for master of parallelized evolution strategies (ES). 
    
    It internally defines an ES algorithm. 
    In each generation, it distributes all sampled solution candidates, each for one worker,
    to compute a list of object function values and then update the ES. 
    
    For more details about how master class works, please refer
    to the documentation of the class, BaseIterativeMaster. 
    
    All inherited subclasses should implement the following function:
    1. make_es(self)
    2. _process_es_result(self, result)
    """
    def __init__(self,
                 num_iteration, 
                 worker_class, 
                 num_worker,
                 init_seed=0, 
                 daemonic_worker=None):
        super().__init__(num_iteration=num_iteration, 
                         worker_class=worker_class, 
                         num_worker=num_worker,
                         init_seed=init_seed, 
                         daemonic_worker=daemonic_worker)
        # Create ES solver
        self.es = self.make_es()
        # It is better to force popsize to be number of workers
        assert self.es.popsize == self.num_worker
        
    def make_es(self):
        """
        User-defined function to create an ES algorithm. 
        
        Returns:
            es (BaseES): An instantiated object of an ES class. 
            
        Examples:
            cmaes = CMAES(mu0=[3]*100, 
                          std0=0.5, 
                          popsize=12)
            return cmaes
        """
        raise NotImplementedError

    def make_tasks(self, iteration):
        # ES samples new candidate solutions
        solutions = self.es.ask()
        
        # Record iteration number, for logging in _process_workers_result()
        # And it also keeps API untouched for assign_tasks() in non-iterative Master class
        self.generation = iteration
        
        return solutions
        
    def _process_workers_result(self, tasks, workers_result):
        # Rename, in ES context, the task is to evalute the solution candidate
        solutions = tasks
        
        # Unpack function values from workers results, [solution_id, function_value]
        # Note that the workers result already sorted ascendingly with respect to task ID
        function_values = [result[1] for result in workers_result]
        
        # Update ES
        self.es.tell(solutions, function_values)
        
        # Obtain results from ES
        result = self.es.result
        
        # Process the ES result
        self._process_es_result(result)
            
    def _process_es_result(self, result):
        """
        User-defined function to process the result from ES. 
        
        Note that the user can use the class memeber `self.generation` which indicate the index of
        the current generation, it is automatically incremented each time when sample a set of
        solution candidates. 
        
        Args:
            result (dict): A dictionary of result returned from es.result. 
            
        Examples:
            best_f_val = result['best_f_val']
            if self.generation == 0 or (self.generation+1) % 100 == 0:
                print(f'Best function value at generation {self.generation+1}: {best_f_val}')
        """
        raise NotImplementedError