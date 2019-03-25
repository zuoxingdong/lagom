import numpy as np


class RunningMeanVar(object):
    r"""Estimates sample mean and variance by using `Chan's method`_. 
    
    It supports for both scalar and multi-dimensional data, however, the input is
    expected to be batched. The first dimension is always treated as batch dimension.
    
    .. note::
    
        For better precision, we handle the data with `np.float64`.
    
    .. warning::
    
        To use estimated moments for standardization, remember to keep the precision `np.float64`
        and calculated as ..math:`\frac{x - \mu}{\sqrt{\sigma^2 + 10^{-8}}}`. 
    
    Example:
    
        >>> f = RunningMeanVar(shape=())
        >>> f([1, 2])
        >>> f([3])
        >>> f([4])
        >>> f.mean
        2.499937501562461
        >>> f.var
        1.2501499923440393
    
    .. _Chan's method:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        
    """
    def __init__(self, shape):
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.N = 1e-4  # or zero ?
        
    def __call__(self, x):
        r"""Update the mean and variance given an additional batched data. 
        
        Args:
            x (object): additional batched data.
        """
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == len(self.shape) + 1
        
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_N = x.shape[0]
        
        new_N = self.N + batch_N
        delta = batch_mean - self.mean
        new_mean = self.mean + delta*(batch_N/new_N)
        M_A = self.N*self.var
        M_B = batch_N*batch_var
        M_X = M_A + M_B + (delta**2)*((self.N*batch_N)/new_N)
        new_var = M_X/new_N
        
        self.mean = new_mean
        self.var = new_var
        self.N = new_N
    
    @property
    def n(self):
        r"""Returns the total number of samples so far. """
        return int(self.N)
