import numpy as np

from .base_transform import BaseTransform


class InterpCurve(BaseTransform):
    r"""Piecewise linear interpolation of a discrete set of data points and generate new :math:`x-y` values
    from the interpolated line. 
    
    .. note::
    
        This is very useful for plotting a set of curves with uncertainty bands where each curve
        has data points at different :math:`x` values. To generate such plot, we need the set of :math:`y` 
        values with consistent :math:`x` values. 
        
    .. note::
    
        Acceptable input data could be either one or two dimensional array. If one dimensional array
        received, then it is treated as a single curve and returns generated points within the min/max
        of horizontal axis in the given data. If two dimensional array received, then it is treated
        as a batch of curves, for horizontal axis, a global min and max are calculated over 
        the entire batch, and then returns the generated data points for each curve with shared
        horizontal axis with the range of global min and global max. 
        
    .. warning::
    
        Piecewise linear interpolation often can lead to more realistic uncertainty bands. Do not
        use polynomial interpolation which the resulting curve can be extremely misleading. 
    
    Example::
    
        x1 = [1, 4, 5, 7, 9, 13, 20]
        y1 = [0.1, 0.25, 0.22, 0.53, 0.37, 0.5, 0.55]
        x2 = [2, 4, 6, 7, 9, 11, 15]
        y2 = [0.03, 0.12, 0.4, 0.2, 0.18, 0.32, 0.39]
        
        >>> plt.plot(x1, y1)
        >>> plt.plot(x2, y2, 'red')
        
        >>> interp = InterpCurve()
        >>> new_x, (new_y1, new_y2) = interp([x1, x2], [y1, y2], num_point=100)
        
        >>> plt.plot(new_x, new_y1)
        >>> plt.plot(new_x, new_y2, 'red')
    
    """
    def __call__(self, x, y, num_point):
        r"""Interpolate the data. 
        
        Args:
            x (ndarray): x values. 
            y (ndarray): y values. 
            num_point (int): number of points to generate from the interpolated line. 
            
        Returns
        -------
        out_x : ndarray
            newly generated horizontal values (shared over entire batch of curves)
        out_y : ndarray
            newly generated vertical values from each interpolated line. 
        """
        assert not np.isscalar(x), 'does not support scalar value !'
        assert not np.isscalar(y), 'does not support scalar value !'
        
        # Convert input to ndarray
        x = self.to_numpy(x, np.float32)
        y = self.to_numpy(y, np.float32)
        assert x.ndim in [1, 2], 'only one or two dimensional data'
        assert y.ndim in [1, 2], 'only one or two dimensional data'
        assert x.ndim == y.ndim, 'x and y should have identical dimensionality'
        
        # wrap 1-d data as a batch
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        if y.ndim == 1:
            y = np.expand_dims(y, 0)
            
        # Get global min and max for x values in the batched data
        min_x = x.min()
        max_x = x.max()
        
        # Generate new query x values between global min and max
        new_x = np.linspace(min_x, max_x, num=num_point)
        
        # Generate new y values from each interpolated line given the shared query x values
        new_y = np.asarray([np.interp(new_x, curve_x, curve_y) for curve_x, curve_y in zip(x, y)])
        new_y = new_y.astype(x.dtype)
        
        return new_x, new_y
