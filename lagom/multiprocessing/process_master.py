# Note that `__name__ == '__main__'` is only required for Windows compatibility
# We don't use it because Ubuntu is expected. 

from abc import ABC
from abc import abstractmethod

from multiprocessing import Process
from multiprocessing import Pipe


class ProcessMaster(ABC):
    r"""Base class for all masters implemented with Python multiprocessing.Process. 
    
    It creates a number of workers each with an individual Process. The communication between master
    and each worker is via independent Pipe connection. The master assigns tasks to workers. When all
    tasks are done, it stops all workers and terminate all processes. 
    
    .. note::
    
        If there are more tasks than workers, then tasks will be splitted into chunks.
        If there are less tasks than workers, then we reduce the number of workers to the number of tasks. 
    
    """
    def __init__(self, worker_class, num_worker):
        self.worker_class = worker_class
        self.num_worker = num_worker
        
    def __call__(self):
        tasks = self.make_tasks()
        if len(tasks) < self.num_worker:
            self.num_worker = len(tasks)
        
        self.make_workers()
        
        results = self.assign_tasks(tasks)
        
        self.close()
        
        return results
        
    def make_workers(self):
        self.master_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.num_worker)])
        
        # daemonic process not allow to have children
        self.list_process = [Process(target=self.worker_class, args=[master_conn, worker_conn], daemon=False)
                             for master_conn, worker_conn in zip(self.master_conns, self.worker_conns)]
        [process.start() for process in self.list_process]
        
        # Not used here. Already copied by forked process
        [worker_conn.close() for worker_conn in self.worker_conns]
        
    @abstractmethod
    def make_tasks(self):
        r"""Returns a list of tasks. 
        
        Returns
        -------
        tasks : list
            a list of tasks
        """
        pass
        
    def assign_tasks(self, tasks):
        r"""Assign a given list of tasks to the workers and return the received results. 
        
        Args:
            tasks (list): a list of tasks
            
        Returns
        -------
        results : object
            received results
        """
        jobs = [[] for _ in range(self.num_worker)]
        for task_id, task in enumerate(tasks):
            jobs[task_id % self.num_worker].append([task_id, task])  # job = [task_id, task]
            
        [master_conn.send(job) for master_conn, job in zip(self.master_conns, jobs)]
        
        results = [None for _ in range(len(tasks))]
        for master_conn in self.master_conns:
            for task_id, result in master_conn.recv():
                results[task_id] = result
        
        return results
    
    def close(self):
        r"""Defines everything required after finishing all the works, e.g. stop all workers, clean up. """
        [master_conn.send('close') for master_conn in self.master_conns]
        assert all([master_conn.recv() == 'confirmed' for master_conn in self.master_conns])
        
        [master_conn.close() for master_conn in self.master_conns]
        assert all([master_conn.closed for master_conn in self.master_conns])
        
        [process.join() for process in self.list_process]
