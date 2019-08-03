import numpy as np
from scipy.signal import lfilter


def geometric_cumsum(alpha, x):
    r"""Calculate future accumulated sums for each element in a list with an exponential factor. 
    
    Given input data :math:`x_1, \dots, x_n` and exponential factor :math:`\alpha\in [0, 1]`, it returns
    an array :math:`y` with the same length and each element is calculated as following
    
    .. math::
        y_i = x_i + \alpha x_{i+1} + \alpha^2 x_{i+2} + \dots + \alpha^{n-i-1}x_{n-1} + \alpha^{n-i}x_{n}
            
    .. note::
        To gain the optimal runtime speed, we use ``scipy.signal.lfilter``
    
    Example:
    
        >>> geometric_cumsum(0.1, [1, 2, 3, 4])
        array([[1.234, 2.34 , 3.4  , 4.   ]])
    
    Args:
        alpha (float): exponential factor between zero and one. 
        x (list): input data
            
    Returns:
        ndarray: calculated data
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = np.expand_dims(x, 0)
    assert x.ndim == 2 
    return lfilter([1], [1, -alpha], x[:, ::-1], axis=1)[:, ::-1]
