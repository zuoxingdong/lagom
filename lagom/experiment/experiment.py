import torch

import numpy as np

from multiprocessing import Process
from multiprocessing import Queue


class BaseExperiment(object):
    def __init__(self, logger=None):
        self.logger = logger
        self.env = self._make_env()
        self.list_configs = self._configure()
        
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
        
    def _make_env(self):
        """
        User-defined environment
        
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
        
        # Shared memory across processes, useful to logger for different configs
        logger_queue = Queue()
        
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

                # Set random seed, e.g. PyTorch, environment, numpy
                self.env.seed(config['seed'])
                torch.manual_seed(config['seed'])
                np.random.seed(config['seed'])

                # Run each algorithm for the specific configuration
                for algo in self.algos:
                    process = Process(target=algo.run, args=[self.env, config, logger_queue])
                    process.start()
                    list_process.append(process)
                
            # Join the processes
            [process.join() for process in list_process]
            
        # Merge all loggers
        self._merge_loggers(logger_queue)
        
    def _merge_loggers(self, logger_queue):
        while not logger_queue.empty():
            # Get logger from the Queue
            logger = logger_queue.get()
            # Initialize dictionary for each algorithm
            if not logger.name in self.logger.logs:
                self.logger.logs[logger.name] = {}
            # Merge logging dictionaries
            self.logger.logs[logger.name] = {**self.logger.logs[logger.name], **logger.logs}