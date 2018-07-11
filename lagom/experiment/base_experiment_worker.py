from lagom.experiment import Config

from lagom.core.multiprocessing import BaseWorker


class BaseExperimentWorker(BaseWorker):
    """
    Base class of the worker for parallelized experiment. 
    
    For details about worker in general, please refer to 
    the documentation of the class, BaseWorker. 
    
    All inherited subclasses should implement the following function:
    1. make_algo(self)
    """
    def work(self, master_cmd):
        task_id, task, seed = master_cmd
        
        # Don't use this seed
        # Set seed inside config
        config = task
        
        # Instantiate an algorithm
        algo = self.make_algo()
        
        # Print configuration
        Config.print_config(config)
        
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