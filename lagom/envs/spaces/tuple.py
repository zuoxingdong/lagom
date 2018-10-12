import numpy as np

from .space import Space


class Tuple(Space):
    r"""A tuple of elementary spaces.
    
    Example::
    
        >>> from lagom.envs.spaces import Discrete, Box, Dict
        >>> Tuple([Discrete(5), Box(-1.0, 1.0, np.float32, shape=(2, 3)), Dict({'success': Discrete(2), 'velocity': Box(-1, 1, np.float32, shape=(1, 3))})])
        Tuple(Discrete(5), Box(2, 3), Dict('success: Discrete(2)', 'velocity: Box(1, 3)'))
        
    """
    def __init__(self, spaces):
        r"""Define a tuple of elementary spaces.
        
        Args:
            spaces (list): a list of elementary spaces. 
        """
        assert isinstance(spaces, (list, tuple))
        self.spaces = tuple(spaces)
        
    def sample(self):
        return tuple([space.sample() for space in self.spaces])
    
    @property
    def flat_dim(self):
        return int(np.sum([space.flat_dim for space in self.spaces]))  # raw int for PyTorch compatibility
    
    def flatten(self, x):
        return np.concatenate([space.flatten(x_part) for x_part, space in zip(x, self.spaces)])
        
    def unflatten(self, x):
        dims = [space.flat_dim for space in self.spaces]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [space.unflatten(flattened) 
                            for flattened, space in zip(list_flattened, self.spaces)]
        
        return tuple(list_unflattened)
    
    def contains(self, x):
        x = tuple(x)
        assert len(x) == len(self.spaces)
        
        return all([x_part in space for x_part, space in zip(x, self.spaces)])
    
    def __repr__(self):
        return f'Tuple({", ".join([str(space) for space in self.spaces])})'
    
    def __eq__(self, x):
        return isinstance(x, Tuple) and tuple(x.spaces) == tuple(self.spaces)
