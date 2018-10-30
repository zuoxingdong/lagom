from abc import ABC
from abc import abstractmethod

from lagom.multiprocessing import MPMaster


class BaseESMaster(MPMaster, ABC):
    r"""Base class for the master of parallelized evolution strategies (ES). 
    
    It internally defines an ES algorithm. 
    For each generation (iteration), it samples a set of solution candidates each assigned to
    an individual :class:`BaseESWorker`. After receiving the objective function values for all
    candidates, the master does an ES update. 
    
    The subclass should implement at least the following:
    
    - :meth:`make_es`
    - :meth:`process_es_result`
    
    """
    def __init__(self, config, worker_class, **kwargs):
        self.config = config
        self.num_generation = config['train.num_iteration']
        
        self.es = self.make_es(self.config)
        
        super().__init__(worker_class, self.es.popsize)
        
        for key, value in kwargs.items():
            self.__setattr__(key, value)
    
    def __call__(self):
        self.make_workers()
        
        for generation in range(self.num_generation):
            self.generation = generation
            
            tasks = self.make_tasks()
            assert len(tasks) == self.num_worker
            
            results = self.assign_tasks(tasks)
            self.process_results(tasks, results)
        
        self.close()
    
    @abstractmethod
    def make_es(self, config):
        r"""Create an ES algorithm. 
        
        Example::
        
            def make_es(self, config):
                es = CMAES(mu0=[3]*100, 
                           std0=0.5, 
                           popsize=12)
                return es
        
        Args:
            config (dict): a dictionary of configurations. 
        
        Returns
        -------
        es : BaseES
            an instantiated object of an ES class. 
        """
        pass

    def make_tasks(self):
        solutions = self.es.ask()
        
        tasks = [[self.config, solution] for solution in solutions]
        
        return tasks
        
    def process_results(self, tasks, results):
        _, solutions = zip(*tasks)
        function_values = results
        
        self.es.tell(solutions, function_values)
        
        result = self.es.result
        self.process_es_result(result)
            
    @abstractmethod
    def process_es_result(self, result):
        r"""Processes the results from one update of the ES. 
        
        Example::
        
            best_f_val = result['best_f_val']
            if self.generation == 0 or (self.generation+1) % 100 == 0:
                print(f'Best function value at generation {self.generation+1}: {best_f_val}')
        
        Args:
            result (object): a result returned from ``es.result``.         
        """
        pass
