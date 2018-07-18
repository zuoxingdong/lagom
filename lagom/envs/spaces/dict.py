import numpy as np

from collections import OrderedDict

from .space import Space


class Dict(Space):
    """
    A dictionary of elementary spaces. 
    
    Simple example:
        Dict({'position': Discrete(2), 'velocity': Discrete(3)})
    Nested example:
        Dict({
            'sensors': Dict({
                'position': Box(low=-10, high=10, shape=(3,), dtype=np.float32),
                'velocity': Box(low=-1, high=1, shape=(3,), dtype=np.float32)
                })
            })
    
    """
    def __init__(self, spaces):
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
        return np.concatenate([self.spaces[key].flatten(item) for key, item in x.items()])
    
    def unflatten(self, x):
        """
        Unflatten the vector into Dict space. Note that the order must be consistent with self.spaces.
        """
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
