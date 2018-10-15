import numpy as np

from .base_transform import BaseTransform


class Clip(BaseTransform):
    r"""Clip the input data with lower and upper bounds. 
    
    Example::
    
        >>> clip = Clip()
        >>> clip(x=[1, 2, 3, 4, 5], a_min=2.5, a_max=3.5)
        array([2.5, 2.5, 3. , 3.5, 3.5], dtype=float32)
    
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
        
        out = np.clip(x, a_min, a_max).astype(x.dtype)
        
        return out
