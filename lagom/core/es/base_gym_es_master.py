from .base_es_master import BaseESMaster


class BaseGymESMaster(BaseESMaster):
    r"""Base class for master of evolution strategies with OpenAI gym environment.
    
    .. note::
    
        It is similar to :class:`BaseESMaster`, the major difference is that this class
        receives an argument :attr:`make_env()` in the constructor and pack it with each 
        task sending to the workers. Each worker can use it to create an environment. 
    
    See :class:`BaseESMaster` for more details about the msater for ES. 
    
    The subclass should implement at least the following:
    
    - :meth:`make_es`
    - :meth:`_process_es_result`
    
    """
    def __init__(self,
                 make_env, 
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
        
        self.make_env = make_env
        
    def make_tasks(self, iteration):
        # Call parent class's method to make tasks (solutions)
        solutions = super().make_tasks(iteration)
        
        # Pack make_env together for each solution
        # e.g. ES with gym environment
        solutions = list(zip(solutions, [self.make_env for _ in range(len(solutions))]))
        assert len(solutions) == self.es.popsize
        
        return solutions
        
    def _process_workers_result(self, tasks, workers_result):
        # Unpack tasks to solutions, it inclues [solutions, make_env]
        solutions = [task[0] for task in tasks]
        
        # Call parent class's method to deal with unpacked solutions
        super()._process_workers_result(tasks=solutions, workers_result=workers_result)
