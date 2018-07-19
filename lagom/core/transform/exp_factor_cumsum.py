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
        
    def __call__(self, x, masks=None):
        """
        Calculate future accumulated sums with exponential factor. 
        
        An option with binary masks is provided. 
        Intuitively, the computation will restart for each occurrence
        of zero. If nothing provided, the default mask is ones everywhere.
        
        Args:
            x (list): input data
            masks (list): binary mask for each data item. 
            
        Returns:
            out (list): calculated data
        """
        # Convert input to ndarray
        x = self.make_input(x)
        # Convert to list
        x = x.tolist()
        
        # Enforce input data as list type
        assert isinstance(x, list), f'Input data must be list dtype, but got {type(x)}. '
        
        # Enforce masks as list type
        if masks is None:
            masks = [1.0]*len(x)
        assert isinstance(masks, list), f'Masks must be list dtype, but got {type(masks)}.'
        assert len(x) == len(masks), 'The length of input data should be the same as the length of masks.'
        assert np.array_equal(masks, np.array(masks).astype(bool)), 'The masks must be binary, i.e. either 0 or 1. '
        
        # buffer of accumulated sum
        cumsum = 0
        
        out = []
        
        # iterate items in reverse ordering
        for val, mask in zip(x[::-1], masks[::-1]):
            cumsum = val + self.alpha*cumsum*mask  # recursive update
            out.insert(0, cumsum)  # insert to the front

        return out
