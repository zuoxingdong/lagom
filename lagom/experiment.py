import torch

import numpy as np

from multiprocessing import Process


class BaseExperiment(object):
    def __init__(self, num_configs, env_settings=None):
        self.num_configs = num_configs
        self.env_settings = env_settings
        
        self.list_args = self._configure(self.num_configs)
        self.env = self._make_env(self.env_settings)
    
    def add_algo(self, algo):
        # TODO: support multiple algorithms
        self.algo = algo
    
    def _configure(self, num_configs):
        """
        Generate all settings and needed set of hyperparameters as list of args (Namespace object)
        
        Args:
            num_configs (int): the number of configurations
            
        Returns:
            list_args (list): list of configurations, each is a Namespace object
        """
        raise NotImplementedError
        
    def _make_env(self, settings=None):
        """
        User-defined environment
        
        Args:
            settings (dict)[optional]: The environment settings
            
        Returns:
            env (Env object): user-defined environment
        """
        raise NotImplementedError
        
    def benchmark(self, num_process=1):
        if num_process > len(self.list_args):
            raise ValueError('The number of process should not be larger than the number of configurations.')   
        
        # Create batches of args to run in parallel with Process
        for i in range(0, len(self.list_args), num_process):
            batch_args = self.list_args[i : i+num_process]
            
            list_process = []
            # Run experiments for the batched args, each with an individual Process
            for args in batch_args:
                print('{:#^50}'.format('#'))
                print('# Job ID: {:<10}'.format(args.ID))
                print('{:#^50}'.format('#'))

                # Set random seed
                self.env.seed(args.seed)
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)

                # Run algorithm for specific configuration
                process = Process(target=self.algo.run, args=[self.env, args])
                process.start()
                
                list_process.append(process)
                
            # Join the processes
            [process.join() for process in list_process]