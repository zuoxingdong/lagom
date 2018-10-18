import numpy as np

from .space import Space


class Dict(Space):
    r"""A dictionary of elementary spaces. 
    
    .. warning::
    
        Make sure to use Python 3.7+, the order-preserving of dict is default. 
    
    There are two common use cases:
    
    * Simple example::
    
        >>> from lagom.envs.spaces import Discrete, Box
        >>> space = Dict({'position': Discrete(2), 'velocity': Box(-1.0, 1.0, np.float32, shape=(1, 2))})
        >>> space
        Dict('position: Discrete(2)', 'velocity: Box(1, 2)')
        >>> space.sample()
        {'position': 0, 'velocity': array([[-0.91862124,  0.08054964]], dtype=float32)}
        
    * Nested example::
    
        >>> sensor_space = Dict({'position': Box(-100, 100, np.float32, shape=(3,)), 'velocity': Box(-1, 1, np.float32, shape=(3,))})
        >>> space = Dict({'sensors': sensor_space, 'score': Discrete(100)})
        >>> space
        Dict("sensors: Dict('position: Box(3,)', 'velocity: Box(3,)')", 'score: Discrete(100)')
        >>> space.sample()
        {'sensors': {'position': array([-43.369488, -24.831665, -22.616478], dtype=float32),
          'velocity': array([0.9381505 , 0.35863736, 0.786181  ], dtype=float32)},
         'score': 83}
    
    """
    def __init__(self, spaces):
        r"""Define a dictionary of elementary spaces. 
        
        Args:
            spaces (dict): a dictionary of elementary spaces. 
        """
        assert isinstance(spaces, dict)
        self.spaces = spaces
        
    def sample(self):
        return {key: space.sample() for key, space in self.spaces.items()}
    
    @property
    def flat_dim(self):
        dim = np.sum([space.flat_dim for space in self.spaces.values()])
        return int(dim)  # raw int for PyTorch compatibility
    
    def flatten(self, x):
        return np.concatenate([self.spaces[key].flatten(item) for key, item in x.items()])
    
    def unflatten(self, x):
        dims = [space.flat_dim for key, space in self.spaces.items()]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [(key, space.unflatten(flattened)) 
                            for flattened, (key, space) in zip(list_flattened, self.spaces.items())]
        
        return dict(list_unflattened)
    
    def contains(self, x):
        assert len(x) == len(self.spaces)
        
        for key, space in self.spaces.items():
            if key not in x:
                return False
            if x[key] not in space:
                return False
        return True
    
    def __repr__(self):
        return f'Dict{tuple([key + ": " + str(space) for key, space in self.spaces.items()])}'
    
    def __eq__(self, x):
        return isinstance(x, Dict) and x.spaces == self.spaces
