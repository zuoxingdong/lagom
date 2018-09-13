import numpy as np

from .space import Space


class Discrete(Space):
    r"""A discrete space in {0, 1, ..., n-1}. 
    
    Example::
    
        >>> Discrete(2)
        
    """
    def __init__(self, n):
        r"""Defines the number of elements in this space. 
        
        Args:
            n (int): the number of elements in this space. 
        """
        assert isinstance(n, int) and n >= 0
        
        self.n = n
        
        super().__init__(None, np.int32)
        
    def sample(self):
        return int(np.random.randint(self.n))
    
    def contains(self, x):
        assert isinstance(x, int), f'expected int dtype, got {type(x)}'
            
        return x >= 0 and x < self.n
    
    @property
    def flat_dim(self):
        return int(self.n)  # PyTorch compatibility
    
    def flatten(self, x):
        # One-hot representation
        onehot = np.zeros(self.n).astype(self.dtype)
        onehot[x] = 1
        return onehot
        
    def unflatten(self, x):
        # Extract index of one-hot representation
        return int(np.nonzero(x)[0][0])
    
    def __repr__(self):
        return f'Discrete({self.n})'
    
    def __eq__(self, x):
        return isinstance(x, Discrete) and x.n == self.n
