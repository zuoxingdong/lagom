from abc import ABC
from abc import abstractmethod


class ProcessWorker(ABC):
    r"""Base class for all workers implemented with Python multiprocessing.Process. 
    
    It communicates with master via a Pipe connection. The worker is stand-by infinitely waiting for task
    from master, working and sending back result. When it receives a ``close`` command, it breaks the infinite
    loop and close the connection. 
        
    """
    def __init__(self, master_conn, worker_conn):
        # Not used here. It's copied by forked process. 
        master_conn.close()
        
        while True:
            job = worker_conn.recv()
            
            if job == 'close':
                worker_conn.send('confirmed')
                worker_conn.close()
                break
            else:
                result = [[task_id, self.work(task_id, task)] for task_id, task in job]
                worker_conn.send(result)
                
    @abstractmethod
    def work(self, task_id, task):
        r"""Work on the given task and return the result. 
        
        Args:
            task_id (int): the task ID.
            task (object): a given task. 
            
        Returns
        -------
        result : object
            working result. 
        """
        pass
