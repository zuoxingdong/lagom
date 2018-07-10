import numpy as np


class BaseTransform(object):
    """
    Base class for transforms, clipping, normalize, centralize, standardize etc. 
    
    Note that all inherited class should support only scalar value or 1-dim vector. 
    Because it has much higher risk to introduce bugs with larger dimensionality. 
    
    It is recommended to convert all numpy processed data to type, np.float32
    becuase it is more compatible with PyTorch. Numpy default float64 often 
    can lead to numerical issues or raised exceptions in PyTorch. Similarly
    for np.int32. 
    """
    def __call__(self, x):
        """
        Process the input data
        
        Args:
            x (scalar/list/ndarray): input data
            
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
            # Convert to type of int32 or float32 with compatibility to PyTorch
            if x.dtype == np.int:
                x = x.astype(np.int32)
            elif x.dtype == np.float:
                x = x.astype(np.float32)

            if x.ndim <= 1:
                return x
            else:
                raise ValueError('Only scalar or 1-dim vector are supported. ')
        else:
            raise TypeError('Only following types are supported: scalar, list, ndarray. ')