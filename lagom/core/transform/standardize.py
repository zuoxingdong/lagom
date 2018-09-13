import numpy as np

from .base_transform import BaseTransform


class Standardize(BaseTransform):
    r"""Standardize the input data to zero mean and standard deviation one.
    
    Each element is subtracted by the mean and divided by standard deviation.
    
    Let :math:`x_1, \dots, x_N` be :math:`N` samples, and let :math:`\mu = \frac{1}{N}\sum_{i=1}^{N} x_i` be
    the sample mean and :math:`\sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N} (x_i - \mu)^2}` be the sample
    standard deviation, then the standardization does the following:
    
    .. math::
        \hat{x}_i = \frac{x_i - \mu}{\sigma}, \forall i\in \{ 1, \dots, N \}
    
    .. warning::
    
        The mean and standard deviation are calculated over the first dimension. This allows to 
        deal with batched data with shape ``[N, ...]`` where ``N`` is the batch size. However, be 
        careful when you want to standardize a multidimensional array with mean/std over all elements. 
        This is not supported.
    
    Example::
    
        >>> standardize = Standardize()
        >>> standardize([0, 5, 10])
        array([-1.2247449,  0.       ,  1.2247449], dtype=float32)
        
        >>> standardize([[1, 2], [3, 2]])
        array([[-0.9999999,  0.       ],
               [ 0.9999999,  0.       ]], dtype=float32)
        
    
    """
    def __init__(self, eps=np.finfo(np.float32).eps):
        r"""Initialize the transform. 
        
        Args:
            eps (float): small positive value to avoid numerical unstable division (zero division).
        """
        self.eps = eps
        
    def __call__(self, x, mean=None, std=None):
        r"""Standardize the input data.
        
        Args:
            x (object): input data
            mean (ndarray): if not ``None``, then use this specific mean to standardize the input. 
            std (ndarray): if not ``None``, then use this specific std to standardize the input. 
        
        Returns
        -------
        out : ndarray
            standardized data
        """
        assert not np.isscalar(x), 'does not support scalar value !'
        
        # Convert input to ndarray
        x = self.to_numpy(x, np.float32)
        
        # Get mean/std
        if mean is None:  # compute the mean
            # keepdims=True very important ! otherwise wrong value
            mean = x.mean(0, keepdims=True)  # over first dimension e.g. batch dim
        if std is None:  # cmopute the std
            # keepdims=True very important ! otherwise wrong value
            std = x.std(0, keepdims=True)  # over first dimension e.g. batch dim
        mean = np.asarray(mean).astype(x.dtype)
        std = np.asarray(std).astype(x.dtype)
        
        # Standardize to unit Gaussian
        out = (x - mean)/(std + self.eps)
        
        return out
