import numpy as np

from .base_transform import BaseTransform


class Centralize(BaseTransform):
    r"""Centralize the input data to zero-centered. 
    
    Let :math:`x_1, \dots, x_N` be :math:`N` samples, the centralization does the following:
    
    .. math::
        \hat{x}_i = x_i - \frac{1}{N}\sum_{j=1}^{N} x_j, \forall i\in \{ 1, \dots, N \}
    
    Example::
    
        >>> centralize = Centralize()
        >>> centralize([1, 2, 3, 4], 0)
        array([-1.5, -0.5,  0.5,  1.5], dtype=float32)
        
        >>> centralize([[1, 3], [2, 4], [3, 5]], 0)
        array([[-1., -1.],
               [ 0.,  0.],
               [ 1.,  1.]], dtype=float32)
               
        >>> mean = [0.1, 0.2, 0.3]
        >>> centralize([1, 2, 3], 0, mean=mean)
        array([0.9, 1.8, 2.7], dtype=float32)
   
    """
    def __call__(self, x, dim, mean=None):
        r"""Centralize the input data. 
        
        Args:
            x (object): input data
            dim (int): the dimension to centralize
            mean (ndarray): if not ``None``, then use this mean to centralize the input. 
            
        Returns
        -------
        out : ndarray
            centralized data
        """
        assert not np.isscalar(x), 'does not support scalar value !'
        
        x = self.to_numpy(x, np.float32)
        
        if mean is None:
            mean = x.mean(dim, keepdims=True)
        else:
            mean = self.to_numpy(mean, np.float32)
            
        out = x - mean
        
        return out
