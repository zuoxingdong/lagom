import numpy as np

from pathlib import Path
import os

from itertools import product

from multiprocessing import Process
from multiprocessing import Pool


class BaseExperiment(object):
    def __init__(self):
        self.list_configs = self._configure()
        
        # Make env for each config
        for config in self.list_configs:
            config['env'] = self._make_env(config)
        
        self.algos = []
    
    def add_algo(self, algo):
        """
        Add an algorithm for benchmarking. 
        All algorithms share the same set of configurations. 
        
        Args:
            algo (Algorithm object): algorithm
        """
        self.algos.append(algo)
    
    def _configure(self):
        """
        Generate all configurations, e.g. hyperparameters and algorithm settings
        
        Returns:
            list_configs (list): list of configurations, each is a Config object
        """
        raise NotImplementedError
        
    def _make_env(self, config):
        """
        User-defined environment for a given configuration
        
        Args:
            config (dict): a configuration
        
        Returns:
            env (Env object): user-defined environment
        """
        raise NotImplementedError
        
    def benchmark(self, num_process=1):
        """
        Parallelized running each algorithm with all configurations. 
        
        Args:
            num_process (int): the number processes to run at a time, each for one configuration. 
                    Note that an individual process opens internally for each algorithm in the list. 
        """
        if num_process > len(self.list_configs):
            raise ValueError('The number of process should not be larger than the number of configurations.')
        
        # Create batches of configs to run in parallel with Process
        for i in range(0, len(self.list_configs), num_process):
            batch_configs = self.list_configs[i : i+num_process]  # slicing can deal with smaller last batch
            
            list_process = []
            # Run experiments for batched configs, an individual process for each configuration and each algorithm
            for config in batch_configs:
                # Print all the configurations
                print(f'{"#":#^50}')
                [print(f'# {key}: {val}') for key, val in config.items()]
                print(f'{"#":#^50}')

                # Run each algorithm for the specific configuration
                for algo in self.algos:
                    process = Process(target=algo.run, args=[config])
                    process.start()
                    list_process.append(process)
                    
            # Wait the processes
            # NOTE: when using shared memory, use Manager().Queue() instead of Queue to avoid deadlock
            [process.join() for process in list_process]
        
    def save_configs(self):
        np.save('logs/experiment_configs', self.list_configs)
            