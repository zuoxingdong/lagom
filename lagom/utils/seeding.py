import random
import numpy as np
import torch


def set_global_seeds(seed):
    r"""Set the seed for generating random numbers.
    
    It sets the following dependencies with the given random seed:
    
    1. PyTorch
    2. Numpy
    3. Python random
    
    Args:
        seed (int): a given seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # both torch and torch.cuda internally


class Seeder(object):
    r"""A random seed generator. 
    
    Given an initial seed, the seeder can be called continuously to sample a single
    or a batch of random seeds. 
    
    .. note::
    
        The seeder creates an independent RandomState to generate random
        numbers. It does not affect the RandomState in ``np.random``. 
    
    Example::
    
        >>> seeder = Seeder(init_seed=0)
        >>> seeder(size=5)
        [209652396, 398764591, 924231285, 1478610112, 441365315]
        
    """
    def __init__(self, init_seed=0):
        r"""Initialize the seeder. 
        
        Args:
            init_seed (int, optional): Initial seed for generating random seeds. Default: ``0``.
        """
        assert isinstance(init_seed, int) and init_seed >= 0, f'expected non-negative integer, got {init_seed}'
        
        self.rng = np.random.RandomState(seed=init_seed)
        
        # Upper bound for sampling new random seeds
        self.max = np.iinfo(np.int32).max
        
    def __call__(self, size=1):
        r"""Return the sampled random seeds according to the given size. 
        
        Args:
            size (int or list): The size of random seeds to sample. 
            
        Returns
        -------
        seeds : list
            a list of sampled random seeds.
        """
        seeds = self.rng.randint(low=0, high=self.max, size=size).tolist()
        
        return seeds
