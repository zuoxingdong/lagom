import numpy as np

from .base_transform import BaseTransform


class Normalize(BaseTransform):
    r"""Normalize the input data to the range :math:`[0, 1]`. 
    
    Each element is subtracted by the minimal element and divided by the the range (maximal - minimal).
    
    Let :math:`x_1, \dots, x_N` be :math:`N` samples, the normalization does the following:
    
    .. math::
        \hat{x}_i = \frac{x_i - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}, \forall i\in \{ 1, \dots, N \}
    
    .. warning::
    
        The min and max are calculated over the first dimension. This allows to deal with batched data with
        shape ``[N, ...]`` where ``N`` is the batch size. However, be careful when you want to normalize
        a multidimensional array with min/max over all elements. This is not supported. 
    
    Example::
    
        >>> normalize = Normalize()
        >>> normalize([1, 2, 3, 4])
        array([0.        , 0.33333334, 0.6666667 , 1.        ], dtype=float32)
        
        >>> normalize([[1, 5], [4, 2]])
        array([[0., 1.],
               [1., 0.]], dtype=float32)
    
    """
    def __init__(self, eps=np.finfo(np.float32).eps):
        r"""Initialize the transform. 
        
        Args:
            eps (float): small positive value to avoid numerical unstable division (zero division).
        """
        self.eps = eps
    
    def __call__(self, x, minimal=None, maximal=None):
        r"""Normalize the input data. 
        
        Args:
            x (object): input data. 
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
            # keepdims=True very important ! otherwise wrong value
            minimal = x.min(0, keepdims=True)  # over first dimension e.g. batch dim
        if maximal is None:
            # keepdims=True very important ! otherwise wrong value
            maximal = x.max(0, keepdims=True)  # over first dimension e.g. batch dim
        minimal = np.asarray(minimal).astype(x.dtype)
        maximal = np.asarray(maximal).astype(x.dtype)
        
        out = (x - minimal)/(maximal - minimal + self.eps)
        
        return out
