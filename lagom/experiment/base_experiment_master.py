from .config import Config

import numpy as np

from lagom.core.multiprocessing import BaseIterativeMaster


class BaseExperimentMaster(BaseIterativeMaster):
    """
    Base class of the master for parallelized experiment. 
    
    For details about master in general, please refer to 
    the documentation of the class, BaseIterativeMaster. 
    
    All inherited subclasses should at least implement the following function:
    1. process_algo_result(self, config, result)
    2. make_configs(self)
    """
    def __init__(self,
                 worker_class, 
                 max_num_worker=None,
                 daemonic_worker=None):
        """
        Args:
            worker_class (BaseWorker): a callable worker class. Note that it is not recommended to 
                send instantiated object of the worker class, but send class instead.
            max_num_worker (int, optional): maximum number of workers. It has following use cases:
                - If None, then number of wokers set to be the total number of configurations. 
                - If number of configurations less than this max bound, then the number of workers
                    will be automatically reduced to the number of configurations.
                - If number of configurations larger than this max bound, then the rest of configurations
                    will be fed in iteratively complying with this max bound. 
                
                Recommended to set to be the same as number of CPU cores, however, it is not necessary.
            daemonic_worker (bool): If True, then set all workers to be daemonic. 
                Because if main process crashes, we should not cause things to hang.
        """
        self.configs = self.make_configs()
        
        # Compute appropriate number of workers to open
        if max_num_worker is None:  # None, then each configuration uses an individual worker
            num_worker = len(self.configs)
        else:  # A max bound is given
            num_worker = min(max_num_worker, len(self.configs))
        
        num_iteration = int(np.ceil(len(self.configs)/num_worker))
        assert len(self.configs) <= num_iteration*num_worker, 'More configurations than capacity. '
        assert len(self.configs) > (num_iteration - 1)*num_worker, 'Too many unused iterations. '
        
        super().__init__(num_iteration=num_iteration, 
                         worker_class=worker_class, 
                         num_worker=num_worker, 
                         init_seed=0,  # Don't use this internal seeder, but set it in configuration
                         daemonic_worker=daemonic_worker)
        
        self.splitted_configs = np.array_split(self.configs, num_iteration)
        for config in self.splitted_configs:
            assert len(config.tolist()) <= num_worker
        
    def make_tasks(self, iteration):
        tasks = self.splitted_configs.pop(0).tolist()
        
        # Print configuration
        [Config.print_config(config) for config in tasks]
        
        return tasks
    
    def _process_workers_result(self, tasks, workers_result):
        for config, (task_id, result) in zip(tasks, workers_result):
            self.process_algo_result(config, result)
            
    def process_algo_result(self, config, result):
        """
        User-defined function to process the result of the execution
        of the algorithm given the configuration. 
        
        Args:
            config (dict): dictionary of configurations. 
            result (object): result of algorithm execution returned from Algorithm.__call__(). 
        """
        raise NotImplementedError
        
    def make_configs(self):
        """
        User-defined function to define all configurations. 
        e.g. hyperparameters and algorithm settings. 
        
        It is recommeded to use Config class, define different
        configurations and call make_configs() to return
        a list of automatically generated all combination
        of configurations. 
        
        Returns:
            configs (list): output from config.make_configs
            
        Examples:
            config = Config()
            
            config.add_item(name='algo', val='RL')
            config.add_item(name='iter', val=30)
            config.add_item(name='hidden_sizes', val=[64, 32, 16])
            config.add_random_eps(name='lr', base=10, low=-6, high=0, num_sample=10)
            config.add_random_continuous(name='values', low=-5, high=5, num_sample=5)
            config.add_random_discrete(name='select', list_val=[43223, 5434, 21314], num_sample=10, replace=True)
            
            configs = config.make_configs()
            
            return configs
        """
        raise NotImplementedError
    
    def save_configs(self, f):
        """
        Save all configurations returned from `make_configs` (a list of dict). 
        
        Args:
            f (str): path to save all configurations
        """
        np.save(f, self.configs)