import torch

from lagom.core.multiprocessing import BaseWorker


class BaseExperimentWorker(BaseWorker):
    """
    Base class of the worker for parallelized experiment. 
    
    For details about worker in general, please refer to 
    the documentation of the class, BaseWorker. 
    
    All inherited subclasses should at least implement the following functions:
    1. make_algo(self)
    """
    def work(self, master_cmd):
        task_id, task, seed = master_cmd
        
        # Don't use this seed
        # Set seed inside config
        config = task
        
        # Assign a GPU card for this task, rolling with total number of GPUs
        # e.g. we have 30 tasks and 5 GPUs, then each GPU will be assigned with 6 tasks
        if 'cuda' in config and config['cuda']:  # if using GPU
            # Get total number of GPUs
            num_gpu = torch.cuda.device_count()
            # Compute which GPU to assign with rolling ID
            device_id = task_id % num_gpu
            # Assign the GPU device for PyTorch
            torch.cuda.set_device(device_id)
        
        # Instantiate an algorithm
        algo = self.make_algo()
        
        # Run the algorithm with given configuration
        result = algo(config)
        
        return task_id, result
    
    def make_algo(self):
        """
        User-defined function to create an algorithm object. 
        
        Returns:
            algo (Algorithm): instantiated algorithm object. 
        """
        raise NotImplementedError
