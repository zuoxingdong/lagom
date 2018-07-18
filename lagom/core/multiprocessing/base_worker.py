class BaseWorker(object):
    """
    Base class of a callable worker to work on a task assigned by the master.
    
    Each calling it stands by with a infinite while loop, waiting for master's command to work
    and it receives Pipe connection ends between master and itself. 
    
    When it receives a 'close' command from master, it close the worker connection and break the loop.
    
    Note that it is a good practice to close the master connection although it is not used
    because the forked process for the worker will anyway copy both connection ends. 
    
    All inherited subclasses should at least implement the following functions:
    1. work(self, master_cmd)
    
    Remark: To be optimally user-friendly, it is highly recommended not to override constructor __init__
        All additional settings for the worker should be sent directly through `master_cmd`. 
        Thus each time the master can create a worker with a Process without passing arguments to constructor. 
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
            master_cmd (list): master's command. [task_id, task, seed]
            
        Returns:
            task_id (int): task ID
            result (object): working result
        """
        raise NotImplementedError
