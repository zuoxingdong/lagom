import numpy as np

from .base_transform import BaseTransform


class Clip(BaseTransform):
    r"""Clip the input data with lower and upper bounds. 
    
    Example::
    
        >>> clip = Clip()
        >>> clip(2.3, 0.5, 1.5)
        1.5
        
        >>> clip([1, 2, 3], 1.5, 2.5)
        array([1.5, 2. , 2.5], dtype=float32)
        
        >>> clip([1, 2, 3], [0, 1, 3.5], [0.5, 2, 9])
        array([0.5, 2. , 3.5], dtype=float32)
    
    """
    def __call__(self, x, a_min, a_max):
        r"""Clip values by min/max bounds. 
        
        Args:
            x (object): input data
            a_min (float/ndarray): minimum value
            a_max (float/ndarray): maximum value
        
        Returns
        -------
        out : ndarray
            clipped data
        """
        x = self.to_numpy(x, np.float32)
        
        out = np.clip(x, a_min, a_max).astype(np.float32)
        
        return out
