from abc import ABC
from abc import abstractmethod

import numpy as np

from itertools import product

from lagom.utils import pickle_dump
from lagom.utils import yaml_dump

from lagom.multiprocessing import MPMaster


class BaseExperimentMaster(MPMaster, ABC):
    r"""Base class for the master of parallelized experiment. 
    
    It firstly makes a list of all possible configurations (from :meth:`make_configs`), each associated
    with a list of random seeds (from :meth:`make_seeds`). Then it distribute the configurations to 
    workers (:class:`BaseExperimentWorker`) to execute the algorithm with specific configuration. 
    
    The total number of jobs is the number of configurations times the number of random seeds.
    
    The subclass should implement at least the following:

    - :meth:`make_configs`
    - :meth:`make_seeds`
    - :meth:`process_results`

    """
    def __init__(self, worker_class, num_worker):
        super().__init__(worker_class, num_worker)
        
        self.configs = self.make_configs()
        self.seeds = self.make_seeds()
    
    def make_tasks(self):
        tasks = list(product(self.configs, self.seeds))
        
        return tasks
        
    @abstractmethod
    def make_configs(self):
        r"""Returns a list of configurations. 
        
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
        pass
        
    @abstractmethod
    def make_seeds(self):
        r"""Returns a list of random seeds.  
        
        .. note::
        
            It is possible to use :class:`Seeder` to sample random seeds or define by hand. 
        
        Returns
        -------
        list_seed : list
            a list of random seeds
        """
        pass
    
    def save_configs(self, f, method='pickle'):
        r"""Save the list of configurations returned from :meth:`make_configs`. 
        
        Args:
            f (str): file path
            method (str): the method to save the list of configuration. Either 'pickle' or 'yaml'
        """
        methods = ['pickle', 'yaml']
        assert method in methods, f'expected {methods}, got {method}'
        
        if method == 'pickle':
            pickle_dump(obj=self.configs, f=f, ext='.pkl')
        elif method == 'yaml':
            yaml_dump(obj=self.configs, f=f, ext='.yml')
