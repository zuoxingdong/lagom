from .base_master import BaseMaster


class BaseIterativeMaster(BaseMaster):
    r"""Base class for iterative version of a callable :class:`BaseMaster`. 
    
    For each call, it has the following iterative procedure:
    
    1. Initialize all workers
    2. Iteratively make tasks and assign each task to a worker to work
    3. Stop all workers
    
    The subclass should implement at least the following:
    
    - :meth:`make_tasks`
    - :meth:`_process_workers_result`
    
    """
    def __init__(self,
                 num_iteration, 
                 worker_class, 
                 num_worker,
                 init_seed=0, 
                 daemonic_worker=None):
        r"""Initialize the iterative master. 
        
        Args:
            num_iteration (int): number of iterative procedures
            worker_class (BaseWorker): a callable worker class. Note that it is not recommended to 
                send instantiated object of the worker class, but send class instead.
            num_worker (int): number of workers. Recommended to be the same as number of CPU cores. 
            init_seed (int): initial seed for the seeder which samples seeds for workers.
            daemonic_worker (bool): If True, then set all workers to be daemonic. 
                Because if main process crashes, we should not cause things to hang.
        """
        super().__init__(worker_class=worker_class, 
                         num_worker=num_worker, 
                         init_seed=init_seed, 
                         daemonic_worker=daemonic_worker)
        
        self.num_iteration = num_iteration
        
    def __call__(self):
        r"""Initialize all the workers, then iteratively make a set of iteration-dependent tasks 
        and assign each task to a worker. 
        
        After all workers finish their jobs with results processed and all iterations are completed, 
        then stop all workers and terminate all processes. 
        """
        # Initialize all workers
        self.initialize_workers()

        # Iteratively make tasks and assign each task to a worker
        for iteration in range(self.num_iteration):
            tasks = self.make_tasks(iteration)
            assert len(tasks) <= self.num_worker, 'The number of tasks cannot exceed the number of workers.'
            self.assign_tasks(tasks)

        # Stop all workers and terminate all processes
        self.stop_workers()
        
    def make_tasks(self, iteration):
        r"""Returns a set of iteration-dependent tasks.
        
        Args:
            iteration (int): the iteration index
            
        Returns
        -------
        tasks : list
            a list of tasks
        """
        raise NotImplementedError
