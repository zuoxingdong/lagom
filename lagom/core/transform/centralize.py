import numpy as np

from .base_transform import BaseTransform


class Centralize(BaseTransform):
    """
    Centralize the input data: Subtracted by mean. 
    """
    def __call__(self, x, mean=None):
        """
        Centralize the input data: Subtracted by mean. 
        
        Args:
            x (scalar/list/ndarray): input data. 
            mean (float): If not None, then use specific mean to centralize the input data. 
        
        Returns:
            out (ndarray): centralized data
        """
        # Convert input to ndarray
        x = self.make_input(x)
        
        # Scalar: return the value
        if x.size == 1:
            return x
        
        # Compute the mean if not provided
        if mean is None:
            mean = x.mean()
            
        # Centralize the data
        out = x - mean
        
        return out.astype(np.float32)
