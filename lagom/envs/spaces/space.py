import numpy as np


class Space(object):
    r"""Base class for all observation and action spaces which could be applied to :class:`Env`.
    
    The subclass should implement at least the following:
    
    - :meth:`sample`
    - :meth:`flat_dim`
    - :meth:`flatten`
    - :meth:`unflatten`
    - :meth:`contains`
    
    """
    def __init__(self, shape=None, dtype=None):
        if shape is None:
            self.shape = None
        else:
            self.shape = tuple(shape)
            
        if dtype is None:
            self.dtype = None
        else:
            self.dtype = np.dtype(dtype)  # create a dtype object
    
    def sample(self):
        r"""Uniformly sample an element from this space."""
        raise NotImplementedError
    
    @property
    def flat_dim(self):
        r"""Return a flattened dimension. """
        raise NotImplementedError
        
    def flatten(self, x):
        r"""Returns the flattened x. """
        raise NotImplementedError
        
    def unflatten(self, x):
        r"""Returns the unflattened x according to defined shape. """
        raise NotImplementedError

    def contains(self, x):
        r"""Return ``True`` if x is contained in this space. """
        raise NotImplementedError
        
    def __contains__(self, x):
        return self.contains(x)
