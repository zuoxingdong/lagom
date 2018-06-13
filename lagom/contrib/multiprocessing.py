import numpy as np

from multiprocessing import Process
from multiprocessing import Pipe


import torch
# SimpleQueue sometimes better, it does not use additional threads
from torch.multiprocessing import SimpleQueue, Queue

# TODO: consider support of torch.multiprocessing or not
# `__name__ == '__main__'` is mostly for Windows compatibility, I don't need it here, as we use Ubuntu in ML


class Seeder(object):
    """
    Define a seeder that can continuously sample a single or a batch of random seeds. 
    """
    def __init__(self, init_seed=0):
        # Initialize random seed for sampling random seeds
        np.random.seed(init_seed)
        # Upper bound of seeds
        self.max = np.iinfo(np.int32).max
        
    def next_seeds(self, size=1):
        return np.random.randint(self.max, size=size).tolist()


class BaseWorker(object):
    """
    Base class of a callable worker to work on a task assigned by the master.
    
    Each calling it stands by with a infinite while loop, waiting for master's command to work
    and it receives Pipe connection ends between master and itself. 
    
    When it receives a 'close' command from master, it close the worker connection and break the loop.
    
    Note that it is a good practice to close the master connection although it is not used
    because the forked process for the worker will anyway copy both connection ends. 
    """
    def __call__(self, master_conn, worker_conn):
        # Close the master connection end as it is not used here
        # The forked process with copy both connections anyway
        master_conn.close()
        
        while True:  # waiting and working for master's command until master say close
            master_cmd = worker_conn.recv()
            
            if master_cmd == 'close':
                worker_conn.close()
                break
            elif master_cmd == 'cozy':
                worker_conn.send('roger')
            else:
                task_id, result = self.work(master_cmd)
                # Send working result back to the master
                # It is important to send task ID, keep track of which task the result belongs to
                worker_conn.send([task_id, result])
        
    def work(self, master_cmd):
        """
        Define how to do the work given the master's command and returns the working result.
        
        Args:
            master_cmd (list): master's command. The first element must contain task_id. 
            
        Returns:
            task_id (int): task ID
            result (object): working result
        """
        raise NotImplementedError
        
        
class BaseMaster(object):
    """
    Base class of a callable master to parallelize solving a set of tasks, each with a worker. 
    
    Each calling it initialize all the workers (each opens a Process) and independent Pipe connections
    between each worker and itself. And then it makes a set of tasks and assign each task to a worker.
    After processing each working results received from workers, it stops all workers and terminate
    all processes. 
    
    Note that it is possible to make less tasks than the number of workers, however, it is not generally
    recommended to do so. 
    """
    def __init__(self, 
                 worker, 
                 num_worker,
                 init_seed=0, 
                 daemonic_worker=None):
        """
        Args:
            worker (BaseWorker): a callable worker class. 
            num_worker (int): number of workers. Recommended to be the same as number of CPU cores. 
            init_seed (int): initial seed for the seeder which samples seeds for workers.
            daemonic_worker (bool): If True, then set all workers to be daemonic. 
                Because if main process crashes, we should not cause things to hang.
        """
        self.worker = worker
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
        self.list_process = [Process(target=self.worker, 
                                     args=[master_conn, worker_conn], 
                                     daemon=self.daemonic_worker) 
                             for master_conn, worker_conn in zip(self.master_conns, self.worker_conns)]
        
        # Start (fork) all processes, so all workers are stand by waiting for master's command to work
        # Note that Linux OS will fork all connection terminals, so it's good to close unused ones here.
        [process.start() for process in self.list_process]
        
        # Close all worker connections here as they are not used in master process
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
        seeds = self.seeder.next_seeds(size=num_task)
        
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
            workers_result (list): a list of restuls from all workers. 
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
    """
    def __init__(self,
                 num_iteration, 
                 worker, 
                 num_worker,
                 init_seed=0, 
                 daemonic_worker=None):
        """
        Args:
            num_iteration (int): number of iterative procedures
            worker (BaseWorker): a callable worker class. 
            num_worker (int): number of workers. Recommended to be the same as number of CPU cores. 
            init_seed (int): initial seed for the seeder which samples seeds for workers.
            daemonic_worker (bool): If True, then set all workers to be daemonic. 
                Because if main process crashes, we should not cause things to hang.
        """
        super().__init__(worker=worker, 
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
        
    def _process_workers_result(self, tasks, workers_result):
        """
        Process the results from all workers. 
        
        Args:
            tasks (list): a list of tasks corresponding to workers results.
            workers_result (list): a list of restuls from all workers. 
        """
        raise NotImplementedError