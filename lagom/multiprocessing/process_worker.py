from abc import ABC

from .base_worker import BaseWorker


class ProcessWorker(BaseWorker, ABC):
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
                result = [[task_id, self.work(task)] for task_id, task in job]
                worker_conn.send(result)
