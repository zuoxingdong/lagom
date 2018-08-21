import torch
import numpy as np
import random


def set_global_seeds(seed):
    r"""Set seed for all dependencies we use. 
    
    It includes the following:
    1. PyTorch
    2. Numpy
    3. Python random
    
    Args:
        seed (int): seed
    """
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Seeder(object):
    r"""A seeder that can continuously sample a single or a batch of random seeds. 
    """
    def __init__(self, init_seed=0):
        # Initialize random seed for sampling random seeds
        np.random.seed(init_seed)
        # Upper bound of seeds
        self.max = np.iinfo(np.int32).max
        
    def __call__(self, size=1):
        return np.random.randint(self.max, size=size).tolist()
