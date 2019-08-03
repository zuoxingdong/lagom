import numpy as np
from scipy.signal import savgol_filter


def smooth_filter(x, window_length, polyorder, **kwargs):
    r"""Smooth a sequence of noisy data points by applying `Savitzky–Golay filter`_. It uses least
    squares to fit a polynomial with a small sliding window and use this polynomial to estimate
    the point in the center of the sliding window. 
    
    This is useful when a curve is highly noisy, smoothing it out leads to better visualization quality.
    
    .. _Savitzky–Golay filter:
        https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    
    Example:
    
        >>> import matplotlib.pyplot as plt
    
        >>> x = np.linspace(0, 4*2*np.pi, num=100)
        >>> y = x*(np.sin(x) + np.random.random(100)*4)
        >>> y2 = smooth_filter(y, window_length=31, polyorder=10)
        
        >>> plt.plot(x, y)
        >>> plt.plot(x, y2, 'red')
        
    Args:
        x (list): one-dimensional vector of scalar data points of a curve. 
        window_length (int): the length of the filter window
        polyorder (int): the order of the polynomial used to fit the samples
        
    Returns:
        ndarray: an numpy array of smoothed curve data
    """
    x = np.asarray(x)
    assert x.ndim == 1, 'only a single vector of scalar values is supported'
    out = savgol_filter(x, window_length, polyorder, **kwargs)
    return out
