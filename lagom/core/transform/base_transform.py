import numpy as np


class BaseTransform(object):
    """
    Base class for transforms, clipping, normalize, centralize, standardize etc. 
    
    Note that all inherited class should support only scalar value or 1-dim vector. 
    Because it has much higher risk to introduce bugs with larger dimensionality. 
    """
    def __call__(self, x):
        """
        Process the input data
        
        Args:
            x: input data
            
        Returns:
            out: The processed data
        """
        raise NotImplementedError
        
    def make_input(self, x):
        """
        Conver the input as scalar or 1-dim ndarray
        
        1. scalar: retain the same
        2. list: convert to 1-dim ndarray with shape [D]
        
        Args:
            x (scalar/list/ndarray): input data
            
        Returns:
            x (ndarray): converted data
        """
        # Enforce tuple becomes list
        if isinstance(x, tuple):
            x = list(x)
        
        if np.isscalar(x) or isinstance(x, (list, np.ndarray)):  # scalar, list or ndarray
            x = np.array(x)

            if x.ndim <= 1:
                return x
            else:
                raise ValueError('Only scalar or 1-dim vector are supported. ')
        else:
            raise TypeError('Only following types are supported: scalar, list, ndarray. ')