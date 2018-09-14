import torch

from lagom.core.multiprocessing import BaseWorker


class BaseExperimentWorker(BaseWorker):
    r"""Base class for the worker of parallelized experiment. 
    
    It executes the algorithm with the configuration and random seed which are distributed by the master. 
    
    .. note::
    
        If the configuration indicates to use GPU (i.e. ``config['cuda']=True``), then each worker will
        assign a specific CUDA device for PyTorch in rolling manner. Concretely, if there are 5 GPUs
        available and the master assigns 30 workers in current iteration, then each GPU will be assigned
        by 6 workers. The GPU is chosen by the worker ID modulus total number of GPUs. In other words, the 
        workers iterate over all GPUs in rolling manner trying to use all GPUs exhaustively for maximal speedup. 
    
    See :class:`BaseWorker` for more details about the workers.
    
    The subclass should implement at least the following:

    - :meth:`make_algo`
    
    """
    def work(self, master_cmd):
        task_id, task, _worker_seed = master_cmd
        # Do not use the worker seed
        # Use seed packed inside task instead
        
        # Unpack task
        config, seed = task
        
        # Assign a GPU card for this task, rolling with total number of GPUs
        # e.g. we have 30 tasks and 5 GPUs, then each GPU will be assigned with 6 tasks
        if 'cuda' in config and config['cuda']:  # if using GPU
            # Get total number of GPUs
            num_gpu = torch.cuda.device_count()
            # Compute which GPU to assign with rolling ID
            device_id = task_id % num_gpu
            # Assign the GPU device in PyTorch
            torch.cuda.set_device(device_id)
            # Create a device string
            device_str = f'cuda:{device_id}'
        else:  # not using CUDA, only CPU
            device_str = 'cpu'
        
        # Instantiate an algorithm
        algo = self.make_algo()
        
        # Run the algorithm with given configuration and seed, and device string
        result = algo(config, seed, device_str=device_str)
        
        return task_id, result
    
    def make_algo(self):
        r"""Returns an instantiated object of an Algorithm class. 
        
        Returns
        -------
        algo : BaseAlgorithm
            an instantiated object of an Algorithm class. 
        """
        raise NotImplementedError
