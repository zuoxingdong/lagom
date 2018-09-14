import numpy as np

from scipy.signal import savgol_filter

from lagom.core.transform import BaseTransform


class SmoothFilter(BaseTransform):
    r"""Smooth a sequence of noisy data points by applying `Savitzky–Golay filter`_. It uses least
    squares to fit a polynomial with a small sliding window and use this polynomial to estimate
    the point in the center of the sliding window. 
    
    This is useful when a curve is highly noisy, smoothing it out leads to better visualization quality.
    
    .. _Savitzky–Golay filter:
        https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    
    Example::
    
        >>> import matplotlib.pyplot as plt
    
        >>> x = np.linspace(0, 4*2*np.pi, num=100)
        >>> y = x*(np.sin(x) + np.random.random(100)*4)
        
        >>> smooth = SmoothFilter()
        >>> y2 = smooth(y, window_length=31, polyorder=10)
        
        >>> plt.plot(x, y)
        >>> plt.plot(x, y2, 'red')
    
    """
    def __call__(self, x, **kwargs):
        r"""Smooth the curve. 
        
        Args:
            x (object): one-dimensional vector of scalar data points of a curve. 
            **kwargs: keyword arguments to specify Savitzky–Golay filter from scipy.
                The required keys are ``[window_length, polyorder]``. 

        Returns
        -------
        out : ndarray
            smoothed curve data
        """
        assert not np.isscalar(x), 'does not support scalar value !'
        assert 'window_length' in kwargs, 'kwargs must contain window_length'
        assert 'polyorder' in kwargs, 'kwargs must contain polyorder'
        
        # Convert input to ndarray
        x = self.to_numpy(x, np.float32)
        
        assert x.ndim == 1, 'only a single vector of scalar values is supported'
        
        # Smooth the curve
        out = savgol_filter(x, **kwargs)
        out = out.astype(np.float32)
        
        return out
