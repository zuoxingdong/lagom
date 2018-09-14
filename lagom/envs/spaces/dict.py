# TODO: on-hold to replace OrderedDict with dict as Python 3.7+ has order-preserving by default. Maybe in 2019/2020.
import numpy as np

from collections import OrderedDict

from .space import Space


class Dict(Space):
    r"""A dictionary of elementary spaces. 
    
    There are two common use cases:
    
    * Simple example::
    
        >>> from lagom.envs.spaces import Discrete, Box
        >>> space = Dict({'position': Discrete(2), 'velocity': Box(low=-1.0, high=1.0, shape=(1, 2), dtype=np.float32)})
        >>> space.sample()
        OrderedDict([('position', 0),
             ('velocity', array([[0.8046695 , 0.78866726]], dtype=float32))])
    
    * Nested example::
    
        >>> sensor_space = Dict({'position': Box(-100, 100, shape=(3,), dtype=np.float32), 'velocity': Box(-1, 1, shape=(3,), dtype=np.float32)})
        >>> space = Dict({'sensors': sensor_space, 'score': Discrete(100)})
        >>> space.sample()
        OrderedDict([('score', 47),
             ('sensors',
              OrderedDict([('position',
                            array([11.511177, 76.35527 , 34.259117], dtype=float32)),
                           ('velocity',
                            array([-0.41881245, -0.85459644,  0.60434735], dtype=float32))]))])
            
    .. note::
    
        From Python 3.7+, the ``dict`` is order-preserving by default so it is recommended
        to use latest Python version. 
    
    """
    def __init__(self, spaces):
        r"""Define a dictionary of elementary spaces. 
        
        Args:
            spaces (dict): a dictionary of elementary spaces. 
        """
        if isinstance(spaces, OrderedDict):
            spaces = spaces
        elif isinstance(spaces, dict):
            spaces = OrderedDict(sorted(list(spaces.items())))
        else:
            raise TypeError('The dtype of input must be either dict or OrderedDict. ')
        self.spaces = spaces
        
        super().__init__(None, None)  # No specific shape and dtype
    
    def sample(self):
        return OrderedDict([(key, space.sample()) for key, space in self.spaces.items()])
    
    def contains(self, x):
        if not isinstance(x, (dict, OrderedDict)) or len(x) != len(self.spaces):
            return False
        for key, space in self.spaces.items():
            if key not in x:
                return False
            if not space.contains(x[key]):
                return False
        return True
    
    @property
    def flat_dim(self):
        dim = np.sum([space.flat_dim for key, space in self.spaces.items()])
        return int(dim)  # PyTorch Tensor dimension only accepts raw int type
    
    def flatten(self, x):
        return np.concatenate([self.spaces[key].flatten(item) for key, item in x.items()]).astype(np.float32)
    
    def unflatten(self, x):
        # Note that the order in x must be consistent with that of in self.spaces
        dims = [space.flat_dim for key, space in self.spaces.items()]
        # Split big vector into a list of vectors for each space
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        # Unflatten for each space
        list_unflattened = [(key, space.unflatten(flattened)) for flattened, (key, space) in zip(list_flattened, self.spaces.items())]
        
        return OrderedDict(list_unflattened)
    
    def __repr__(self):
        return f'Dict{tuple([key + ": " + str(space) for key, space in self.spaces.items()])}'
    
    def __eq__(self, x):
        return isinstance(x, Dict) and x.spaces == self.spaces
