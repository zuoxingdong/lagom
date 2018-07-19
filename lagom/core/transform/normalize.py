import numpy as np

from .base_transform import BaseTransform


class Normalize(BaseTransform):
    """
    Normalize the input data: Subtracted by minimal and divided by range (maximal - minimal)
    """
    def __init__(self, eps=np.finfo(np.float32).eps):
        self.eps = eps
    
    def __call__(self, x, min_val=None, max_val=None):
        """
        Normalize the input data: Subtracted by minimal and divided by range (maximal - minimal)
        
        Args:
            x (object): input data. 
            min_val (float): If not None, then use specific min values to normalize the input. 
            max_val (float): If not None, then use specific max values to normalize the input. 
            
        Returns:
            out (ndarray): normalized data
        """
        # Convert input to ndarray
        x = self.make_input(x)
        
        # Scalar: return clipped value
        if x.size == 1:
            return np.clip(x, a_min=0, a_max=1)
        
        # Compute min/max if they are not provided
        if min_val is None or max_val is None:
            min_val = x.min()
            max_val = x.max()
            
        # Normalize data into [0, 1]
        out = (x - min_val)/(max_val - min_val + self.eps)
        
        return out.astype(np.float32).tolist()  # enforce raw float dtype
