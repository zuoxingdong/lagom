import numpy as np

from .space import Space


class Box(Space):
    r"""A continuous space in :math:`R^n`. Each dimension is bounded by low/high. """
    def __init__(self, low, high, dtype, shape=None):
        r"""Define the lower and upper bound for this space. 
        
        There are two common use cases:
            
        * Identical bound for each dimension::

            >>> Box(low=-1.0, high=2.0, dtype=np.float32, shape=[3, 4])
            Box(3, 4)
            
        * Independent bound for each dimension::
        
            >>> Box(low=np.array([-1.0, -2.0]), high=np.array([3.0, 4.0]), dtype=np.float32)
            Box(2,)
        
        """
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = np.dtype(dtype)
        
        if shape is None:
            assert low.shape == high.shape
            self.shape = low.shape
            
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.shape = tuple(shape)
            
            self.low = np.full(self.shape, low)
            self.high = np.full(self.shape, high)
        
        self.low = self.low.astype(self.dtype)
        self.high = self.high.astype(self.dtype)
        
    def sample(self):
        return np.random.uniform(low=self.low, high=self.high, size=self.shape).astype(self.dtype)
    
    @property
    def flat_dim(self):
        return int(np.prod(self.shape))  # raw int for PyTorch compatibility
    
    def flatten(self, x):
        return np.asarray(x).flatten()
    
    def unflatten(self, x):
        return np.asarray(x).reshape(self.shape)
    
    def contains(self, x):
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)
    
    def __repr__(self):
        return f'Box{self.shape}'
    
    def __eq__(self, x):
        return isinstance(x, Box) and np.allclose(x.low, self.low) and np.allclose(x.high, self.high)
