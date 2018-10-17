import numpy as np

from .base_transform import BaseTransform


class Normalize(BaseTransform):
    r"""Normalize the input data to the range :math:`[0, 1]`. 
    
    Each element is subtracted by the minimal element and divided by the the range (maximal - minimal).
    
    Let :math:`x_1, \dots, x_N` be :math:`N` samples, the normalization does the following:
    
    .. math::
        \hat{x}_i = \frac{x_i - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}, \forall i\in \{ 1, \dots, N \}
    
    Example::
    
        >>> normalize = Normalize()
        >>> normalize([1, 2, 3, 4], 0)
        array([0.        , 0.33333334, 0.6666667 , 1.        ], dtype=float32)
        
        >>> normalize([[1, 5], [4, 2]], 0)
        array([[0., 1.],
               [1., 0.]], dtype=float32)
    
    """
    def __init__(self, eps=np.finfo(np.float32).eps):
        r"""Initialize the transform. 
        
        Args:
            eps (float): small positive value to avoid numerical unstable division (zero division).
        """
        self.eps = eps
    
    def __call__(self, x, dim, minimal=None, maximal=None):
        r"""Normalize the input data. 
        
        Args:
            x (object): input data. 
            dim (int): the dimension to normalize
            minimal (ndarray): If not ``None``, then use this specific min to normalize the input. 
            maximal (ndarray): If not ``None``, then use this specific max to normalize the input. 
            
        Returns
        -------
        out : ndarray
            normalized data
        """
        assert not np.isscalar(x), 'does not support scalar value !'
        
        x = self.to_numpy(x, np.float32)
        
        if minimal is None:
            minimal = x.min(dim, keepdims=True)
        else:
            minimal = self.to_numpy(minimal, np.float32)
            
        if maximal is None:
            maximal = x.max(dim, keepdims=True)
        else:
            maximal = self.to_numpy(maximal, np.float32)
        
        out = (x - minimal)/(maximal - minimal + self.eps)
        
        return out
