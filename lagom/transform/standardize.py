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
    
    Example::
    
        >>> standardize = Standardize()
        >>> standardize([0, 5, 10], 0)
        array([-1.2247449,  0.       ,  1.2247449], dtype=float32)
        
        >>> standardize([[1, 2], [4, 6], [3, 7]], 0)
        array([[-1.3363061 , -1.3887302 ],
               [ 1.0690447 ,  0.46291006],
               [ 0.26726115,  0.9258201 ]], dtype=float32)
               
        >>> standardize([[1, 2], [4, 6], [3, 7]], 1)
        array([[-0.99999976,  0.99999976],
               [-0.9999999 ,  0.9999999 ],
               [-1.        ,  1.        ]], dtype=float32)
    
    """
    def __init__(self, eps=np.finfo(np.float32).eps):
        r"""Initialize the transform. 
        
        Args:
            eps (float): small positive value to avoid numerical unstable division (zero division).
        """
        self.eps = eps
        
    def __call__(self, x, dim, mean=None, std=None):
        r"""Standardize the input data.
        
        Args:
            x (object): input data
            dim (int): the dimension to standardize
            mean (ndarray): if not ``None``, then use this specific mean to standardize the input. 
            std (ndarray): if not ``None``, then use this specific std to standardize the input. 
        
        Returns
        -------
        out : ndarray
            standardized data
        """
        assert not np.isscalar(x), 'does not support scalar value !'
        
        x = self.to_numpy(x, np.float32)
        
        if mean is None:
            mean = x.mean(dim, keepdims=True)
        else:
            mean = self.to_numpy(mean, np.float32)
            
        if std is None:
            std = x.std(dim, keepdims=True)
        else:
            std = self.to_numpy(std, np.float32)
        
        out = (x - mean)/(std + self.eps)
        
        return out
