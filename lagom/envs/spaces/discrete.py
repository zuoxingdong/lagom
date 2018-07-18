import numpy as np

from .space import Space


class Discrete(Space):
    """
    A discrete space in {0, 1, ..., n-1}. 
    """
    def __init__(self, n):
        """
        Define the number of elements in the space. 
        
        Examples:
            Discrete(5)
        """
        self.n = n
        
        super().__init__(None, int)
        
    def sample(self):
        return np.random.randint(self.n)
    
    def contains(self, x):
        if not isinstance(x, int):
            raise TypeError('The input must be of type int. ')
            
        return x >= 0 and x < self.n
    
    @property
    def flat_dim(self):
        return self.n
    
    def flatten(self, x):
        # One-hot representation
        onehot = np.zeros(self.n)
        onehot[x] = 1
        return onehot
        
    def unflatten(self, x):
        # Extract index of one-hot representation
        return np.nonzero(x)[0][0]
    
    def __repr__(self):
        return f'Discrete({self.n})'
    
    def __eq__(self, x):
        return isinstance(x, Discrete) and x.n == self.n
