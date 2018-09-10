from .configurator import Configurator

import numpy as np

from itertools import product

from lagom import pickle_dump
from lagom import yaml_dump

from lagom.core.multiprocessing import BaseIterativeMaster


class BaseExperimentMaster(BaseIterativeMaster):
    r"""Base class for the master of parallelized experiment. 
    
    It firstly makes a list of all possible configurations (from :meth:`make_configs`), each associated
    with a list of random seeds (from :meth:`make_seeds`). Then it distribute the configurations to 
    workers (:class:`BaseExperimentWorker`) to execute the algorithm with specific configuration. 
    
    The total number of jobs is the number of configurations times the number of random seeds.
    
    .. note::
    
        If the number of configurations is larger than the number of workers, then the jobs are splitted
        into iterations, each iteration has the maximum capacity as the number of workers. 
    
    See :class:`BaseIterativeMaster` for more details about the iterative master.
    
    The subclass should implement at least the following:

    - :meth:`make_configs`
    - :meth:`make_seeds`
    - :meth:`process_algo_result`

    """
    def __init__(self,
                 worker_class, 
                 max_num_worker=None,
                 daemonic_worker=None):
        r"""Initialize the experiment master. 
        
        Args:
            worker_class (BaseExperimentWorker): a class of the type of :class:`BaseExperimentWorker`. Note
                that it is the class itself, not an instantiated object. 
            max_num_worker (int, optional): maximum number of workers. It has following use cases:
                * If ``None``, then number of wokers set to be the total number of configurations
                  times the number of random seeds. 
                * If number of configurations less than this max bound, then the number of workers
                  reduces to the number of configurations.
                * If number of configurations larger than this max bound, then the number of workers
                  equals to this max bound. The rest of configurations are splitted into batches, 
                  and run them iteratively with this max bound.
            daemonic_worker (bool): If ``True``, then set all workers to be daemonic. The reason to 
                use daemonic process is because if main process crashes, we should not cause things to hang.
        """
        # Make all configurations and random seeds
        self.configs = self.make_configs()
        self.seeds = self.make_seeds()
        
        # Make all tasks: each configuration with all random seeds
        all_task = list(product(self.configs, self.seeds))
        num_task = len(all_task)
        assert num_task == len(self.configs)*len(self.seeds)
        # Compute number of workers to open
        if max_num_worker is None:  # open individual worker for each task
            num_worker = num_task
        else:  # with max bound
            num_worker = min(max_num_worker, num_task)
        
        # Compute number of iterations
        num_iteration = int(np.ceil(num_task/num_worker))
        assert (num_iteration - 1)*num_worker < num_task, 'Too many unused iterations'
        assert num_iteration*num_worker >= num_task, 'Capacity smaler than number of tasks'
        
        # Call super class to initialize everything
        super().__init__(num_iteration=num_iteration, 
                         worker_class=worker_class, 
                         num_worker=num_worker, 
                         init_seed=0,  # Don't use this internal seeder, but set it in configuration
                         daemonic_worker=daemonic_worker)
        
        # Split all the tasks into batches according to the number of iterations
        self.batch_task = np.array_split(all_task, num_iteration)
        self.batch_task = [batch.tolist() for batch in self.batch_task]
        assert len(self.batch_task) == self.num_iteration
        assert all([len(batch) <= self.num_worker for batch in self.batch_task])
        
    def make_tasks(self, iteration):
        tasks = self.batch_task[iteration]
        
        # Print configuration
        for config, seed in tasks:
            print(f'@ Seed for following configuration: {seed}')
            Configurator.print_config(config)
        
        return tasks
    
    def _process_workers_result(self, tasks, workers_result):
        for (config, seed), (task_id, result) in zip(tasks, workers_result):
            self.process_algo_result(config, seed, result)
            
    def process_algo_result(self, config, seed, result):
        r"""Process the result returned from algorithm for a specific configuration. 
        
        Args:
            config (dict): a dictionary of configurations. 
            seed (int): the seed used for running the algorithm with the given configuration. 
            result (object): result returned from the algorithm.
        """
        raise NotImplementedError
        
    def make_configs(self):
        r"""Returns a list of configurations, each is a dictionary.
        
        .. note::
        
            It is recommended to use :class:`Configurator` to automatically generate
            all configurations either by grid search or by random search. 
            
        .. warning::
        
            One should not specify random seeds here, use :meth:`make_seeds` instead. 
        
        Returns
        -------
        list_config : list
            a list of configurations. 
        """
        raise NotImplementedError
        
    def make_seeds(self):
        r"""Returns a list of random seeds, each associated with a configuration. 
        
        .. note::
        
            It is possible to use :class:`Seeder` to sample random seeds or define by hand. 
        
        Returns
        -------
        list_seed : list
            a list of random seeds
        """
        raise NotImplementedError
    
    def save_configs(self, f, method='pickle'):
        r"""Save the list of configurations returned from :meth:`make_configs`. 
        
        Args:
            f (str): file path
            method (str): the method to save the list of configuration. Either 'pickle' or 'yaml'
        """
        assert isinstance(method, str)
        methods = ['pickle', 'yaml']
        assert method in methods, f'expected {methods}, got {method}'
        
        if method == 'pickle':
            pickle_dump(obj=self.configs, f=f, ext='.pkl')
        elif method == 'yaml':
            yaml_dump(obj=self.configs, f=f, ext='.yml')
