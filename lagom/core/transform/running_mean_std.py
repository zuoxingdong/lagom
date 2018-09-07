import numpy as np

from .base_transform import BaseTransform


class RunningMeanStd(BaseTransform):
    r"""Online algorithm for estimating sample mean and standard deviation
    
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, dtype='list'):
        self.mean = None
        self.var = None
        self.N = 0
        
        # dtype to output mu and sigma
        # Possible: ['list', 'ndarray']
        self.dtype = dtype
        
    def __call__(self, x):
        r"""Update the mean and variance given an additional data. 
        
        Note that the additional data is supported for both scalar, vector or multidimensional array
        For scalar: a single scalar value
        For vector: a sequence of scalar values. 
        For multidimensional array: first dimension is batch size, the batch regarded as a sequence. 
        """
        # Make input as batched shape
        x = self.make_input(x)
        
        # Compute mean and variance over the batch size
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_N = x.shape[0]
        
        # Compute the updated mean, variance
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
        
    def make_input(self, x):
        # Enforce ndarray type
        x = np.array(x)
        
        if x.ndim == 0:  # scalar value: convert to shape [1, 1]
            x = x.reshape([1, 1])
        elif x.ndim == 1:  # vector: sequence of values, add scalar data dimension [N, 1] forming a batch
            x = np.expand_dims(x, axis=1)
            
        return x

    @property
    def mu(self):
        r"""Running mean"""
        if self.dtype == 'list':
            return self.mean.squeeze().astype(np.float32).tolist()  # enforce raw float dtype
        elif self.dtype == 'ndarray':
            return self.mean.squeeze().astype(np.float32)
        
    @property
    def sigma(self):
        r"""Running standard deviation"""
        if self.dtype == 'list':
            return np.sqrt(self.var).squeeze().astype(np.float32).tolist()  # enforce raw float dtype
        elif self.dtype == 'ndarray':
            return np.sqrt(self.var).squeeze().astype(np.float32)
    
    @property
    def n(self):
        r"""Number of samples"""
        return self.N
