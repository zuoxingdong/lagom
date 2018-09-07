import numpy as np

from .base_transform import BaseTransform


class ExpFactorCumSum(BaseTransform):
    r"""Calculate future accumulated sums with exponential factor. 
    
    Given input :math:`[x_1, ..., x_n]` and factor :math:`\alpha`, the computation returns an array y 
    with same length and 
    
    .. math::
        y_i = x_i + \alpha*x_{i+1} + \alpha^2*x_{i+2} + \dots + \alpha^{n-i-1}*x_{n-1} + \alpha^{n-i}*x_{n}
    
    Commonly useful for calculating returns in RL. 
    """
    def __init__(self, alpha):
        r"""Initialize
        
        Args:
            alpha (float): exponential factor
        """
        self.alpha = alpha
        
    def __call__(self, x, mask=None):
        r"""Calculate future accumulated sums with exponential factor. 
        
        An option with binary mask is provided. 
        Intuitively, the computation will restart for each occurrence
        of zero. If nothing provided, the default mask is ones everywhere.
        
        Args:
            x (list): input data
            mask (list): binary mask for each data item. 
            
        Returns
        -------
        out : list
            calculated data
        """
        # Convert input to ndarray
        x = self.make_input(x)
        # Convert to list
        x = x.tolist()
        
        # Enforce input data as list type
        assert isinstance(x, list), f'Input data must be list dtype, but got {type(x)}. '
        
        # Enforce mask as list type
        if mask is None:
            mask = [1.0]*len(x)
        else:  # check is mask is binary array, because boolean array might lead to bugs easily
            assert np.array(mask).dtype != bool, 'Ensure using binary value only, becuase boolean might lead to bugs. '
        assert isinstance(mask, list), f'Mask must be list dtype, but got {type(mask)}.'
        assert len(x) == len(mask), 'The length of input data should be the same as the length of mask.'
        assert np.array_equal(mask, np.array(mask).astype(bool)), 'The mask must be binary, i.e. either 0 or 1. '
        
        # buffer of accumulated sum
        cumsum = 0
        
        out = []
        
        # iterate items in reverse ordering
        for val, mask_item in zip(x[::-1], mask[::-1]):
            cumsum = val + self.alpha*cumsum*mask_item  # recursive update
            out.insert(0, cumsum)  # insert to the front

        return out
