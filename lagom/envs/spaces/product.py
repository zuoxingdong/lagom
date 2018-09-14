import numpy as np

from .space import Space


class Product(Space):
    r"""A product (tuple) of elementary spaces.
    
    Example::
    
        >>> from lagom.envs.spaces import Discrete, Box
        >>> Product((Discrete(5), Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32)))
        
    """
    def __init__(self, spaces):
        r"""Define a product of elementary spaces.
        
        Args:
            spaces (list): a list of elementary spaces. 
        """
        self.spaces = tuple(spaces)
        
        super().__init__(None, None)
        
    def sample(self):
        return tuple([space.sample() for space in self.spaces])
    
    def contains(self, x):
        x = tuple(x)  # ensure tuple type
        
        return all([space.contains(x_part) for x_part, space in zip(x, self.spaces)])
    
    @property
    def flat_dim(self):
        return int(np.sum([space.flat_dim for space in self.spaces]))  # PyTorch compatibility
    
    def flatten(self, x):
        return np.concatenate([space.flatten(x_part) for x_part, space in zip(x, self.spaces)])
        
    def unflatten(self, x):
        dims = [space.flat_dim for space in self.spaces]
        # Split big vector into list of vectors for each space
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        # Unflatten for each space
        list_unflattened = [space.unflatten(flattened) for flattened, space in zip(list_flattened, self.spaces)]
        
        return tuple(list_unflattened)
        
    def __repr__(self):
        return f'Product({", ".join([str(space) for space in self.spaces])})'
    
    def __eq__(self, x):
        return isinstance(x, Product) and tuple(x.spaces) == tuple(self.spaces)
