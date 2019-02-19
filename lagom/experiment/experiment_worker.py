import torch

from lagom.multiprocessing import ProcessWorker


class ExperimentWorker(ProcessWorker):
    r"""The worker of parallelized experiment. 
    
    It assigns a PyTorch device object to each received task according to their ID. 
    
    .. note::
    
        If the configuration indicates to use GPU (i.e. ``config['cuda']=True``), then each worker will
        assign a specific CUDA device for PyTorch in rolling manner. Concretely, if there are 5 GPUs
        available and the master assigns 30 workers, then each GPU will be assigned by 6 workers. 
        The GPU is chosen by the worker ID modulus total number of GPUs. In other words, the 
        workers iterate over all GPUs in rolling manner trying to use all GPUs exhaustively for maximal speedup. 
    
    """    
    def work(self, task_id, task):
        config, seed, run = task
        device = self.make_device(config, task_id)
        
        print(f'@ Seed for following configuration: {seed}')
        print('#'*50)
        [print(f'# {key}: {value}') for key, value in config.items()]
        print('#'*50)
        
        result = run(config=config, seed=seed, device=device)
        
        return result
    
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
