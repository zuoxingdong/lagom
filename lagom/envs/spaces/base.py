import numpy as np


class Space(object):
    """
    Base class for observation and action space applied to general Env.
    """
    def __init__(self, shape):
        if shape is None:
            self.shape = None
        else:
            self.shape = tuple(shape)
            
    @property
    def flat_dim(self):
        """
        Return a flattened dimension
        """
        raise NotImplementedError
        
    def flatten(self, x):
        """
        Return the flattened x.
        """
        raise NotImplementedError
        
    def unflatten(self, x):
        """
        Return the unflattened x according to defined shape
        """
        raise NotImplementedError
            
    def sample(self):
        """
        Uniformly sample an element from this space.
        """
        raise NotImplementedError
        
    def contains(self, x):
        """
        Return True if x is contained in this space.
        """
        raise NotImplementedError