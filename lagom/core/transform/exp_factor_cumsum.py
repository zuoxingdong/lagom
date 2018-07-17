import numpy as np

from .base_transform import BaseTransform


class ExpFactorCumSum(BaseTransform):
    """
    Calculate future accumulated sums with exponential factor. 
    
    e.g. Given input [x_1, ..., x_n] and factor \alpha, the computation returns an array y with same length
    and y_i = x_i + \alpha*x_{i+1} + \alpha^2*x_{i+2} + ... + \alpha^{n-i-1}*x_{n-1} + \alpha^{n-i}*x_{n}
    
    Commonly useful for calculating returns in RL. 
    """
    def __init__(self, alpha):
        """
        Args:
            alpha (float): exponential factor
        """
        self.alpha = alpha
        
    def __call__(self, x):
        """
        Calculate future accumulated sums with exponential factor. 
        
        Args:
            x (list): input data
            
        Returns:
            out (list): calculated data
        """
        # Convert input to ndarray
        x = self.make_input(x)
        # Convert to list
        x = x.tolist()
        
        # Enforce input data as list type
        assert isinstance(x, list), 'Supported type: list. '
        
        # buffer of accumulated sum
        cumsum = 0
        
        out = []
        
        for val in x[::-1]:  # iterate items in reverse ordering
            cumsum = val + self.alpha*cumsum  # recursive update
            out.insert(0, cumsum)  # insert to the front

        return out
