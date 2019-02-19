from abc import ABC
from abc import abstractmethod

from lagom.multiprocessing import ProcessMaster
from lagom import Logger


class BaseESMaster(ProcessMaster, ABC):
    r"""Base class for the master of parallelized evolution strategies (ES). 
    
    For each generation (iteration), it samples a set of solution candidates each assigned to
    an individual :class:`BaseESWorker`. After receiving the objective function values for all
    candidates, the master does an ES update. 
    
    """
    def __init__(self, worker_class, es, config, **kwargs):
        self.es = es
        self.config = config
        self.logger = Logger()
        super().__init__(worker_class, self.es.popsize)
        
        for key, value in kwargs.items():
            self.__setattr__(key, value)
    
    def __call__(self, num_generation):
        self.make_workers()
        
        for generation in range(num_generation):
            tasks = self.make_tasks()
            function_values = self.assign_tasks(tasks)
            solutions = [solution for _, solution in tasks]
            self.es.tell(solutions, function_values)
            self.logging(self.logger, generation, solutions, function_values)
        
        self.close()
        
        return self.logger
    
    def make_tasks(self):
        solutions = self.es.ask()
        tasks = [[self.config, solution] for solution in solutions]
        assert len(tasks) == self.num_worker
        
        return tasks
        
    @abstractmethod
    def logging(self, logger, generation, solutions, function_values):
        r"""Logging for current ES generation. 
        
        Example::
        
            best_f_val = result['best_f_val']
            if self.generation == 0 or (self.generation+1) % 100 == 0:
                print(f'Best function value at generation {self.generation+1}: {best_f_val}')
        
        Args:
            logger (Logger): logger
            generation (int): number of generations
            solutions (list): list of candidate solutions
            function_values (list): list of fitness values for each candidate solution
            
        Returns
        -------
        logger : Logger
        """
        pass
