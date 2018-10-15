import numpy as np

from .base_transform import BaseTransform


class RunningMeanStd(BaseTransform):
    r"""Estimates sample mean and standard deviation by using `Chan's method`_. 
    
    .. note::
    
        It supports for both scalar and high-dimensional data. If input data is a scalar
        value or a single vector of values, then it would be treated as a batch with the
        batch size :math:`1`. Otherwise, the first dimension will always be treated as
        batch dimension. Note that there is an ambiguity of a single vector of values, it
        can be either a batch of scalars or a single high-dimensional data, here it is treated
        as a batch of scalars, so for the latter case, the user has to add a batch dimension
        before feeding such data. 
    
    .. warning::
    
        The internal batched mean and variance are calculated over the first dimension. This allows
        to deal with batched data with shape ``[N, ...]`` where ``N`` is the batch size.
    
    Example::
    
        >>> runningavg = RunningMeanStd()
        >>> [runningavg(i) for i in [1, 2, 3, 4]]
        >>> runningavg.mu
        array(2.5, dtype=float32)
        
        >>> runningavg.sigma
        array(1.118034, dtype=float32)
        
        >>> runningavg = RunningMeanStd()
        >>> runningavg([[1, 4], [2, 3], [3, 2], [4, 1]])
        >>> runningavg.mu
        array([2.5, 2.5], dtype=float32)
        
        >>> runningavg.sigma
        array([1.118034, 1.118034], dtype=float32)
    
    .. _Chan's method:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        
    """
    def __init__(self):
        self.mean = None
        self.var = None
        self.shape = None  # maintain the identical shape
        self.N = 0
        
    def __call__(self, x):
        r"""Update the mean and variance given an additional data. 
        
        Args:
            x (object): additional data to update the estimation of mean and standard deviation. 
        """
        # Convert input to ndarray
        x = self.to_numpy(x, np.float32)
        
        # Store the original data shape, useful for returning mu and sigma with idential shape
        # Only for first time that data comes in, so assume all data flow with same shape
        if self.shape is None:
            if x.ndim == 0 or x.ndim == 1:  # scalar or vector of scalars
                self.shape = ()
            else:  # >=2 dimensions, first dimension is batch dimension
                self.shape = x.shape[1:]
        
        # Make data as batch
        if x.ndim == 0:  # scalar: to shape [1, 1]
            x = x.reshape([1, 1])
        elif x.ndim == 1:  # single vector: to shape [N, 1]
            x = np.expand_dims(x, axis=1)
        assert x.ndim >= 2
        
        # Compute batch mean and variance over first dimension
        # Keep the original dimension
        batch_mean = x.mean(0, keepdims=True)
        batch_var = x.var(0, keepdims=True)
        batch_N = x.shape[0]
        
        # Update mean and variance
        if self.mean is None or self.var is None:  # Initialize mean and variance
            new_mean = batch_mean
            new_var = batch_var
            new_N = batch_N
        else:  # apply the formula
            new_N = self.N + batch_N
            delta = batch_mean - self.mean
        
            new_mean = self.mean + delta*(batch_N/new_N)
        
            M_A = self.N*self.var
            M_B = batch_N*batch_var
            M_X = M_A + M_B + (delta**2)*((self.N*batch_N)/new_N)
        
            new_var = M_X/(new_N)
        
        # Update new values
        self.mean = new_mean
        self.var = new_var
        self.N = new_N

    @property
    def mu(self):
        r"""Returns the current running mean. """
        if self.mean is None:
            return None
        else:
            return self.mean.reshape(self.shape)
        
    @property
    def sigma(self):
        r"""Returns the current running standard deviation. """
        if self.var is None:
            return None
        else:
            return np.sqrt(self.var).reshape(self.shape)
    
    @property
    def n(self):
        r"""Returns the total number of samples so far. """
        return self.N
