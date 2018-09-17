class BaseWorker(object):
    r"""Base class of all callable worker to work on a task assigned by the master.
    
    For each calling, the worker will stand-by in an infinite loop waiting for command from master to do
    the working and send back working result. 
    
    If the master's command is 'close', then it breaks the infinite loop and the connections will be
    closed. 
    
    The subclass should implement at least the following:
    
    - :meth:`prepare`
    - :meth:`work`
    
    .. note::
    
        It is highly discouraged to override the constructor ``__init__``. Because the master will create
        a worker with a Process without passing any argument to the constructor. All additional settings 
        for the worker should be sent directly through ``master_cmd``
        
    """
    def __call__(self, master_conn, worker_conn):
        r"""Stand-by infinitely waiting for master's command and do the work until a 'close'
        is received. 
        
        Args:
            master_conn (Pipe.Connection): Pipe connection from the master side
            worker_conn (Pipe.Connection): Pipe connection for the worker side
        """
        # Close the master connection end as it is not used here
        # The forked process with copy both connections anyway
        master_conn.close()
        
        # Prepare everything before worker official setup
        self.prepare()
        
        while True:  # waiting and working for master's command until master say close
            master_cmd = worker_conn.recv()
            
            if master_cmd == 'close':
                worker_conn.send('confirmed')
                worker_conn.close()
                break
            elif master_cmd == 'cozy':  # no work to do
                worker_conn.send('roger')
            else:
                task_id, result = self.work(master_cmd)
                # Send working result back to the master
                # It is important to send task ID, keep track of which task the result belongs to
                worker_conn.send([task_id, result])
        
    def prepare(self):
        r"""Define the preparations of the worker before it enters infinite loop of standby, listening
        to the master and working until receiving terminal command from master. 
        
        .. note::
        
            This could be useful to define costly and common components for the working. For example
            it could be an agent and an environment which the agent is evaluated on, it can be expensive
            to do it every time the worker gets a new job to work, instead, they can be created only once
            and simply reset for each new task. See :class:`BaseIterativeMaster` for more details. 
            
        .. note::
        
            All created components should be saved as class members. If there is nothing to prepare, then
            the subclass of this class could simply ``pass``. 
        
        """
        raise NotImplementedError
        
    def work(self, master_cmd):
        r"""Define how to do the work given the master's command and returns the working result.
        
        Args:
            master_cmd (list): master's command. [task_id, task, seed]
            
        Returns
        -------
        task_id : int
            task ID
        result : object
            working result
        """
        raise NotImplementedError
