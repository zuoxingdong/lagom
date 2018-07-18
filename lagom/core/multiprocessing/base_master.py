# Note that `__name__ == '__main__'` is mostly for Windows compatibility
# We do not use it because most expected users shall use Ubuntu. 

from multiprocessing import Process
from multiprocessing import Pipe

# TODO: consider support of torch.multiprocessing, for its own SimpleQueue or Queue
"""
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
# SimpleQueue sometimes better, it does not use additional threads
from torch.multiprocessing import SimpleQueue
"""

from .seeder import Seeder


class BaseMaster(object):
    """
    Base class of a callable master to parallelize solving a set of tasks, each with a worker. 
    
    Each calling it initialize all the workers (each opens a Process) and independent Pipe connections
    between each worker and itself. And then it makes a set of tasks and assign each task to a worker.
    After processing each working results received from workers, it stops all workers and terminate
    all processes. 
    
    Note that it is possible to make less tasks than the number of workers, however, it is not generally
    recommended to do so. 
    
    All inherited subclasses should at least implement the following functions:
    1. make_tasks(self)
    2. _process_workers_result(self, tasks, workers_result)
    """
    def __init__(self, 
                 worker_class, 
                 num_worker,
                 init_seed=0, 
                 daemonic_worker=None):
        """
        Args:
            worker_class (BaseWorker): a callable worker class. Note that it is not recommended to 
                send instantiated object of the worker class, but send class instead. 
            num_worker (int): number of workers. Recommended to be the same as number of CPU cores. 
            init_seed (int): initial seed for the seeder which samples seeds for workers.
            daemonic_worker (bool): If True, then set all workers to be daemonic. 
                Because if main process crashes, we should not cause things to hang.
        """
        self.worker_class = worker_class
        self.num_worker = num_worker
        self.init_seed = init_seed
        self.daemonic_worker = daemonic_worker
        
        # Create a seeder, sampling different seeds for each task
        self.seeder = Seeder(init_seed=self.init_seed)
        
    def __call__(self):
        """
        It initializes the workers, makes a set of tasks and assign each task to a worker. 
        After finish processing results from all workers, stop them and terminate all processes. 
        """
        # Initialize all workers
        self.initialize_workers()
        
        # Make tasks and assign each task to a worker
        tasks = self.make_tasks()
        assert len(tasks) <= self.num_worker, 'The number of tasks cannot exceed the number of workers.'
        self.assign_tasks(tasks)
        
        # Stop all workers and terminate all processes
        self.stop_workers()
        
    def initialize_workers(self):
        """
        Initialize all workers, each opens a Process. 
        Create an independent Pipe connection between master and each worker. 
        """
        # Create pipes as communicators between master and workers
        self.master_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.num_worker)])
        
        # Create a Process for each worker
        self.list_process = [Process(target=self.worker_class(),  # individual instantiation for each Process
                                     args=[master_conn, worker_conn], 
                                     daemon=self.daemonic_worker) 
                             for master_conn, worker_conn in zip(self.master_conns, self.worker_conns)]
        
        # Start (fork) all processes, so all workers are stand by waiting for master's command to work
        # Note that Linux OS will fork all connection terminals, so it's good to close unused ones here.
        [process.start() for process in self.list_process]
        
        # Close all worker connections here as they are not used in master process
        # Note that this should come after all the processes started
        [worker_conn.close() for worker_conn in self.worker_conns]
        
    def make_tasks(self):
        """
        Returns a set of tasks.
        
        Returns:
            tasks (list): a list of tasks
        """
        raise NotImplementedError
        
    def assign_tasks(self, tasks):
        """
        Assign each task to a worker. And process the results from all tasks. 
        
        Args:
            tasks (list): a list of tasks
        """
        num_task = len(tasks)
        
        # Sample random seeds, each for one task
        seeds = self.seeder(size=num_task)
        
        # Special case when there are less tasks than number of workers
        if num_task < self.num_worker:
            master_conns = self.master_conns[:num_task]  # slice exact connection terminals
            # Send 'cozy' signal to rest of workers as they don't have work to do
            [master_conn.send('cozy') for master_conn in self.master_conns[num_task:]]
        else:  # same number of tasks and workers
            master_conns = self.master_conns
        
        # Iterate over all tasks, each assigned to a worker to work
        for task_id, (task, seed, master_conn) in enumerate(zip(tasks, seeds, master_conns)):
            # Send the task to worker via master connection
            # It is important to send ID to make received results with consistent order
            master_cmd = [task_id, task, seed]
            master_conn.send(master_cmd)
        
        # Receive results from all workers
        workers_result = [master_conn.recv() for master_conn in self.master_conns]
        assert len(workers_result) == self.num_worker
        # Remove all 'roger' reply from cozy workers
        workers_result = [result for result in workers_result if result != 'roger']
        
        # Reordering the result to be consistent with tasks as workers might finish in different speed
        # i.e. ascending ordering of task ID [0, ..., num_task - 1]
        # each result with data structure: [task_id, result]
        workers_result = sorted(workers_result, key=lambda x: x[0])  # x[0] get task_id
        
        # Process the results from all workers
        self._process_workers_result(tasks, workers_result)
    
    def _process_workers_result(self, tasks, workers_result):
        """
        Process the results from all workers. 
        
        Args:
            tasks (list): a list of tasks corresponding to workers results.
            workers_result (list): a list of restuls from all workers. Each result consists of [task_ID, result]
        """
        raise NotImplementedError
    
    def stop_workers(self):
        """
        Stop all the workers by sending a 'close' signal via pipe connection and join all processes.
        """
        # Tell all workers to stop working and close pipe connections
        for master_conn in self.master_conns:
            master_conn.send('close')
            master_conn.close()
            
        # Join all processes
        [process.join() for process in self.list_process]
