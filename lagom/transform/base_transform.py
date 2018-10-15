from abc import ABC
from abc import abstractmethod

import numpy as np


class BaseTransform(ABC):
    r"""Base class for all transformations e.g. clipping, normalize, centralize, standardize etc. 
    
    .. note::
    
        All transformation should be handled with numpy as much as possible and return results as list. 
        For PyTorch compatibility, integer dtype should be ``np.int32`` and float should be ``np.float32``.
    
    The subclass should implement at least the following:
    
    - :meth:`__call__`
    
    """
    @abstractmethod
    def __call__(self, x):
        r"""Transform the input data. 
        
        Args:
            x (object): input data
            
        Returns
        -------
        out : object
            the processed data
        """
        pass
        
    def to_numpy(self, x, dtype):
        r"""Converts the input data to numpy dtype with PyTorch compatibility. 
        
        Args:
            x (object): input data
            dtype (dtype): PyTorch compatible dtype, e.g. ``np.int32``, ``np.float32``.
            
        Returns
        -------
        out : ndarray
            converted data with numpy dtype
        """
        x = np.asarray(x)
        x = x.astype(dtype)
        
        return x
