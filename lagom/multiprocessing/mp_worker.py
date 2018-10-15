from abc import ABC

from .base_worker import BaseWorker


class MPWorker(BaseWorker, ABC):
    r"""Base class for all workers implemented with Process from Python multiprocessing library. 
    
    It communicates with master via a Pipe connection. The worker is stand-by infinitely waiting for task
    from master, working and sending back result. When it receives a ``close`` command, it breaks the infinite
    loop and close the connection. 
    
    The subclass should implement at least the following:
    
    - :meth:`prepare`
    - :meth:`work`
        
    """
    def __call__(self, master_conn, worker_conn):
        # Not used here. It's copied by forked process. 
        master_conn.close()
        
        self.prepare()
        
        while True:
            task = worker_conn.recv()
            
            if task == 'close':
                worker_conn.send('confirmed')
                worker_conn.close()
                break
            else:
                result = self.work(task)
                worker_conn.send(result)
