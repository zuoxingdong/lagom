import numpy as np

from .base_transform import BaseTransform


class Standardize(BaseTransform):
    """
    Standardize the input data: Subtracted by mean and divided by standard deviation
    """
    def __init__(self, eps=np.finfo(np.float32).eps):
        self.eps = eps
        
    def __call__(self, x, mean=None, std=None):
        """
        Standardize the input data: Subtracted by mean and divided by standard deviation. 
        
        Args:
            x (object): input data
            mean (float): If not None, then use specific mean to standardize the input data. 
            std (float): If not None, then use specific standard deviation to standardize the input data. 
        
        Returns:
            out (ndarray): standardized data
        """
        # Convert input to ndarray
        x = self.make_input(x)
        
        # Scalar: return the value
        if x.size == 1:
            return x
        
        # Compute mean and std if not provided
        if mean is None or std is None:
            mean = x.mean()
            std = x.std()
            
        # Standardize the data
        out = (x - mean)/(std + self.eps)
        
        return out.astype(np.float32)