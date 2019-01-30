from abc import ABC
from abc import abstractmethod

import torch

from lagom.multiprocessing import MPWorker

from .configurator import Configurator


class BaseExperimentWorker(MPWorker, ABC):
    r"""Base class for the worker of parallelized experiment. 
    
    It executes the algorithm with a configuration, a random seed and PyTorch device distributed by the master. 
    
    .. note::
    
        If the configuration indicates to use GPU (i.e. ``config['cuda']=True``), then each worker will
        assign a specific CUDA device for PyTorch in rolling manner. Concretely, if there are 5 GPUs
        available and the master assigns 30 workers in current iteration, then each GPU will be assigned
        by 6 workers. The GPU is chosen by the worker ID modulus total number of GPUs. In other words, the 
        workers iterate over all GPUs in rolling manner trying to use all GPUs exhaustively for maximal speedup. 
    
    See :class:`MPWorker` for more details about the workers.
    
    The subclass should implement at least the following:

    - :meth:`prepare`
    - :meth:`make_algo`
    
    """
    def work(self, task):
        task_id, task, use_chunk = task
        
        if use_chunk:
            results = []
            for one_task in task:
                _, result = self.do_one_task(task_id, one_task)
                results.append(result)
                
            return task_id, results
        else:
            return self.do_one_task(task_id, task)

    def do_one_task(self, task_id, task):
        config, seed = task
        device = self.make_device(config, task_id)
        algo = self.make_algo()

        print(f'@ Seed for following configuration: {seed}')
        Configurator.print_config(config)

        result = algo(config, seed, device)

        return task_id, result
    
    def make_device(self, config, task_id):
        if 'cuda' in config and config['cuda']:
            if 'cuda_ids' in config:  # use specific GPUs
                device_id = config['cuda_ids'][task_id % len(config['cuda_ids'])]
            else:  # use all GPUs
                num_gpu = torch.cuda.device_count()
                device_id = task_id % num_gpu
            
            torch.cuda.set_device(device_id)
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cpu')
            
        return device
    
    @abstractmethod
    def make_algo(self):
        r"""Returns an instantiated object of an Algorithm class. 
        
        Returns
        -------
        algo : BaseAlgorithm
            an instantiated object of an Algorithm class. 
        """
        pass
