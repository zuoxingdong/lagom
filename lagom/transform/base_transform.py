from abc import ABC
from abc import abstractmethod

import numpy as np


class BaseTransform(ABC):
    r"""Base class for all transformations. 
    
    Transformation basically processes the input data. Examples are clipping, standardization, smoothing etc. 
    
    .. note::
    
        The internal computation should be handled with numpy or standard libraries for efficiency and easy maintenance. 
        For PyTorch compatibility, data type should be ``np.int32`` and ``np.float32`` for integers and floats.
    
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
            transformed input data
        """
        pass
        
    def to_numpy(self, x, dtype):
        r"""Converts the input data to numpy dtype for PyTorch compatibility. 
        
        Args:
            x (object): input data
            dtype (dtype): PyTorch compatible dtype, e.g. ``np.int32``, ``np.float32``.
            
        Returns
        -------
        out : ndarray
            converted input data.
        """
        x = np.asarray(x)
        x = x.astype(dtype)
        
        return x
