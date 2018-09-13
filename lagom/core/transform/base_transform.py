import numpy as np


class BaseTransform(object):
    r"""Base class for all transformations e.g. clipping, normalize, centralize, standardize etc. 
    
    .. note::
    
        All numpy processed data should be of dtype, np.int32 or np.float32, for PyTorch compatibility. 
    
    The subclass should implement at least the following:
    
    - :meth:`__call__`
    
    """
    def __call__(self, x):
        r"""Transform the input data. 
        
        Args:
            x (object): input data
            
        Returns
        -------
        out : object
            the processed data
        """
        raise NotImplementedError
        
    def to_numpy(self, x, dtype):
        r"""Converts the input data to numpy dtype with PyTorch compatibility. 
        
        Args:
            x (object): input data
            dtype (dtype): data type with PyTorch compatibility, e.g. np.int32, np.float32
            
        Returns
        -------
        out : ndarray
            converted data with numpy dtype
        """
        # Make array type
        x = np.asarray(x)
        # Convert dtype
        x = x.astype(dtype)
        
        return x
