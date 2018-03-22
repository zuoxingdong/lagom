import numpy as np

from lagom.envs.spaces.base import Space


class Box(Space):
    """
    A continuous space in R^n. Each dimension is bounded by low/high.
    """
    def __init__(self, low, high, dtype, shape=None):
        """
        Define the bound for the space. 
        
        Two cases:
            1. Identical bound for each dimension: 
                Box(low=-1.0, high=1.0, dtype=np.float32, shape=(2, 3))
            2. Separate bound for each dimension: 
                Box(low=np.array([-1.0, -2.0]), high=np.array([3.0, 4.0]), dtype=np.float32)
        """
        if shape is None:  # Case 2
            assert low.shape == high.shape
            
            shape = low.shape
            
            self.low = low
            self.high = high
        else:  # Case 1
            assert np.isscalar(low) and np.isscalar(high)
            
            self.low = np.full(shape, low)
            self.high = np.full(shape, high)
            
        # Ensure dtype
        self.low = self.low.astype(dtype)
        self.high = self.high.astype(dtype)
            
        super().__init__(dtype, shape)
        
    def sample(self):
        return np.random.uniform(low=self.low, high=self.high, size=self.shape).astype(self.dtype)
    
    def contains(self, x):
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)
    
    @property
    def flat_dim(self):
        return int(np.prod(self.shape))  # PyTorch Tensor dimension only accepts raw int type
    
    def flatten(self, x):
        return np.asarray(x).flatten()
    
    def unflatten(self, x):
        return np.asarray(x).reshape(self.shape)
    
    def __repr__(self):
        return f'Box{self.shape}'
    
    def __eq__(self, x):
        return isinstance(x, Box) and np.allclose(x.low, self.low) and np.allclose(x.high, self.high)