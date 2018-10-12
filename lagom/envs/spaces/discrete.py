import numpy as np

from .space import Space


class Discrete(Space):
    r"""A discrete space in :math:`\{ 0, 1, \dots, n-1 \}`. 
    
    Example::
    
        >>> Discrete(2)
        
    """
    def __init__(self, n):
        r"""Define the number of elements in this space. 
        
        Args:
            n (int): the number of elements in this space. 
        """
        self.dtype = np.int32
        
        assert isinstance(n, int) and n >= 0
        self.n = n
        
    def sample(self):
        return int(np.random.randint(self.n))  # raw int for PyTorch compatibility
    
    @property
    def flat_dim(self):
        return int(self.n)  # raw int for PyTorch compatibility
    
    def flatten(self, x):
        # One-hot representation
        onehot = np.zeros(self.n)
        onehot[x] = 1.0
        return onehot
        
    def unflatten(self, x):
        # Extract index from one-hot representation
        return int(np.nonzero(x)[0][0])
    
    def contains(self, x):
        assert isinstance(x, int), f'expected int dtype, got {type(x)}'
            
        return x >= 0 and x < self.n
    
    def __repr__(self):
        return f'Discrete({self.n})'
    
    def __eq__(self, x):
        return isinstance(x, Discrete) and x.n == self.n
