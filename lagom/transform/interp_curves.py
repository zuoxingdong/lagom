import numpy as np


def interp_curves(x, y):
    r"""Piecewise linear interpolation of a discrete set of data points and generate new :math:`x-y` values
    from the interpolated line. 
    
    It receives a batch of curves with :math:`x-y` values, a global min and max of the x-axis are 
    calculated over the entire batch and new x-axis values are generated to be applied to the interpolation
    function. Each interpolated curve will share the same values in x-axis. 
    
    .. note::
    
        This is useful for plotting a set of curves with uncertainty bands where each curve
        has data points at different :math:`x` values. To generate such plot, we need the set of :math:`y` 
        values with consistent :math:`x` values. 
        
    .. warning::
    
        Piecewise linear interpolation often can lead to more realistic uncertainty bands. Do not
        use polynomial interpolation which the resulting curve can be extremely misleading. 
    
    Example::
    
        >>> import matplotlib.pyplot as plt
    
        >>> x1 = [4, 5, 7, 13, 20]
        >>> y1 = [0.25, 0.22, 0.53, 0.37, 0.55]
        >>> x2 = [2, 4, 6, 7, 9, 11, 15]
        >>> y2 = [0.03, 0.12, 0.4, 0.2, 0.18, 0.32, 0.39]
        
        >>> plt.scatter(x1, y1, c='blue')
        >>> plt.scatter(x2, y2, c='red')
        
        >>> new_x, new_y = interp_curves([x1, x2], [y1, y2], num_point=100)
        >>> plt.plot(new_x[0], new_y[0], 'blue')
        >>> plt.plot(new_x[1], new_y[1], 'red')
        
    Args:
            x (list): a batch of x values. 
            y (list): a batch of y values. 
            num_point (int): number of points to generate from the interpolated line. 
    
    Returns:
        tuple: a tuple of two lists. A list of interpolated x values (shared for the batch of curves)
            and followed by a list of interpolated y values. 
    """
    new_x = np.unique(np.hstack(x))
    assert new_x.ndim == 1
    ys = [np.interp(new_x, curve_x, curve_y) for curve_x, curve_y in zip(x, y)]
    return new_x, ys
