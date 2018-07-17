import numpy as np


class Seeder(object):
    """
    A seeder that can continuously sample a single or a batch of random seeds. 
    """
    def __init__(self, init_seed=0):
        # Initialize random seed for sampling random seeds
        np.random.seed(init_seed)
        # Upper bound of seeds
        self.max = np.iinfo(np.int32).max
        
    def __call__(self, size=1):
        return np.random.randint(self.max, size=size).tolist()
