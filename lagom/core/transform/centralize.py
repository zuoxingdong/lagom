import numpy as np

from .base_transform import BaseTransform


class Centralize(BaseTransform):
    r"""Centralize the input data to zero-centered. 
    
    Let :math:`x_1, \dots, x_N` be :math:`N` samples, the centralization does the following:
    
    .. math::
        \hat{x}_i = x_i - \frac{1}{N}\sum_{j=1}^{N} x_j, \forall i\in \{ 1, \dots, N \}
    
    .. warning::
    
        The mean is calculated over the first dimension. This allows to deal with batched data with
        shape ``[N, ...]`` where ``N`` is the batch size. However, be careful when you want to centralize
        a multidimensional array with mean over all elements. This is not supported. 
    
    Example::
    
        >>> centralize = Centralize()
        >>> centralize([1, 2, 3, 4])
        array([-1.5, -0.5,  0.5,  1.5], dtype=float32)
        
        >>> centralize([[1, 3], [2, 11]])
        array([[-0.5, -4. ],
               [ 0.5,  4. ]], dtype=float32)
   
    """
    def __call__(self, x, mean=None):
        r"""Centralize the input data. 
        
        Args:
            x (object): input data
            mean (ndarray): If not ``None``, then use this specific mean to centralize the input. 
            
        Returns
        -------
        out : ndarray
            centralized data
        """
        assert not np.isscalar(x), 'does not support scalar value !'
        
        # Convert input to ndarray
        x = self.to_numpy(x, np.float32)
        
        # Get mean
        if mean is None:  # compute the mean
            # keepdims=True very important ! otherwise wrong value
            mean = x.mean(0, keepdims=True)  # over first dimension e.g. batch dim
        else:  # use provided mean
            mean = np.asarray(mean).astype(x.dtype)
            
        # Centralize the data
        out = x - mean
        
        return out
