from .base_master import BaseMaster


class BaseIterativeMaster(BaseMaster):
    """
    Base class for iterative version of a callable master. 
    It supports iterative procedure during each call as following
    
    # Initialize all workers
    self.initialize_workers()
    
    # Iteratively make and assign tasks
    for iteration in range(num_iterations):
        self.make_tasks(iteration)
        self.assign_tasks()
    
    # Stop all workers and terminate all processes
    self.stop_workers()
    
    All inherited subclasses should implement the following function
    1. make_tasks(self, iteration)
    2. _process_workers_result(self, tasks, workers_result)
    """
    def __init__(self,
                 num_iteration, 
                 worker_class, 
                 num_worker,
                 init_seed=0, 
                 daemonic_worker=None):
        """
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
        """
        It initializes the workers and then iteratively makes a set of iteration-dependent tasks 
        and assign each task to a worker. 
        After processing results from all workers and iterations, stop them and terminate all processes. 
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
        """
        Returns a set of iteration-dependent tasks.
        
        Args:
            iteration (int): the iteration index
            
        Returns:
            tasks (list): a list of tasks
        """
        raise NotImplementedError