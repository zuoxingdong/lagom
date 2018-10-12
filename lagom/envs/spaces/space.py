from abc import ABC
from abc import abstractmethod


class Space(ABC):
    r"""Base class for all observation and action spaces applied to :class:`Env`.
    
    The subclass should implement at least the following:
    
    - :meth:`sample`
    - :meth:`flat_dim`
    - :meth:`flatten`
    - :meth:`unflatten`
    - :meth:`contains`
    
    """
    @abstractmethod
    def sample(self):
        r"""Uniformly sample an element from this space. """
        pass
    
    @property
    @abstractmethod
    def flat_dim(self):
        r"""Return a flattened dimension. """
        pass
        
    @abstractmethod
    def flatten(self, x):
        r"""Returns the flattened x. """
        pass
        
    @abstractmethod
    def unflatten(self, x):
        r"""Returns the unflattened x according to defined shape. """
        pass

    @abstractmethod
    def contains(self, x):
        r"""Return ``True`` if x is contained in this space. """
        pass
        
    def __contains__(self, x):
        return self.contains(x)
