import torch
import numpy as np
import random


def set_global_seeds(seed):
    r"""Set the seed for generating random numbers.
    
    It sets the following dependencies with the given random seed:
    
    1. PyTorch
    2. Numpy
    3. Python random
    
    Args:
        seed (int): Given seed.
    """
    torch.manual_seed(seed)  # both torch and torch.cuda internally
    np.random.seed(seed)
    random.seed(seed)


class Seeder(object):
    r"""A random seed generator. 
    
    Given an initial seed, the seeder can be called continuously to sample a single
    or a batch of random seeds. 
    """
    
    def __init__(self, init_seed=0):
        r"""Constructor.
        
        Args:
            init_seed (int, optional): Initial seed for generating random seeds.
        """
        assert isinstance(init_seed, int) and init_seed >= 0, f'Seed expected to be non-negative integer, got {init_seed}'
        
        # Create a numpy RandomState with given initial seed
        # A RandomState is independent of np.random
        self.rng = np.random.RandomState(seed=init_seed)
        # Upper bound for sampling new random seeds
        self.max = np.iinfo(np.int32).max
        
    def __call__(self, size=1):
        r"""Return the sampled random seeds according to the given size. 
        
        Args:
            size (int or list): The size of random seeds to sample. 
            
        Returns:
            seeds (list): A list of sampled random seeds.
        """
        seeds = self.rng.randint(low=0, high=self.max, size=size).tolist()
        
        return seeds
