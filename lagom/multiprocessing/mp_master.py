# Note that `__name__ == '__main__'` is only required for Windows compatibility
# We don't use it because Ubuntu is expected. 

from abc import ABC

from operator import itemgetter
from itertools import chain

import numpy as np

from multiprocessing import Process
from multiprocessing import Pipe

from .base_master import BaseMaster


class MPMaster(BaseMaster, ABC):
    r"""Base class for all masters implemented with Process from Python multiprocessing library. 
    
    It creates a number of workers each with an individual Process. The communication between master
    and each worker is via independent Pipe connection. The master assigns tasks to workers. When all
    tasks are accomplished, it stops all workers and terminate all processes. 
    
    .. note::
    
        If there are more tasks than workers, then tasks will be splitted into chunks.
        If there are less tasks than workers, then only create workers with the same number of tasks. 
    
    The subclass should implement at least the following:
    
    - :meth:`make_tasks`
    - :meth:`process_results`
    
    """
    def __init__(self, worker_class, num_worker):
        r""" Initialize the master. 
        
        Args:
            worker_class (MPWorker): a callable worker class.
            num_worker (int): number of workers. 
        """
        self.worker_class = worker_class
        self.num_worker = num_worker
        
    def __call__(self):
        tasks = self.make_tasks()
        if len(tasks) < self.num_worker:
            self.num_worker = len(tasks)
        
        self.make_workers()
        
        results = self.assign_tasks(tasks)
        self.process_results(results)
        
        self.close()
        
    def make_workers(self):
        self.master_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.num_worker)])
        
        self.list_process = [Process(target=self.worker_class(),
                                     args=[master_conn, worker_conn], 
                                     daemon=False)  # daemonic process not allow to have children
                             for master_conn, worker_conn in zip(self.master_conns, self.worker_conns)]
        
        [process.start() for process in self.list_process]
        
        # Not used here. Already copied by forked process above
        [worker_conn.close() for worker_conn in self.worker_conns]
        
    def assign_tasks(self, tasks):
        if len(tasks) > self.num_worker:
            use_chunk = True
            list_idx = np.array_split(range(len(tasks)), self.num_worker)
            tasks = [itemgetter(*idx)(tasks) for idx in list_idx]
            tasks = [[task] if idx.size == 1 else task for idx, task in zip(list_idx, tasks)]
        elif len(tasks) == self.num_worker:
            use_chunk = False
        
        for task_id, (task, master_conn) in enumerate(zip(tasks, self.master_conns)):
            master_conn.send([task_id, task, use_chunk])
            
        results = [master_conn.recv() for master_conn in self.master_conns]
        # reordering (ascending) w.r.t. task_id because tasks might finish at different speed
        # format: [task_id, result]
        results = sorted(results, key=lambda x: x[0])
        results = [result[1] for result in results]
        if use_chunk:
            results = list(chain.from_iterable(results))
        
        return results
    
    def close(self):
        [master_conn.send('close') for master_conn in self.master_conns]
        
        close_check = np.all([master_conn.recv() == 'confirmed' for master_conn in self.master_conns])
        assert close_check, 'Something wrong with closing all workers'
        
        [master_conn.close() for master_conn in self.master_conns]
        assert np.all([master_conn.closed for master_conn in self.master_conns]), 'Not all master connections are closed'
        
        [process.join() for process in self.list_process]
