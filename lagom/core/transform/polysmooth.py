import numpy as np

from lagom.core.transform import BaseTransform


class PolySmooth(BaseTransform):
    """
    Use least squares polynomial fit to smooth the curve. 
    """
    def __call__(self, data, poly_deg=10):
        """
        Args:
            data (list/ndarray): one-dimensional vector of data points of a curve. 
            poly_deg (int): Degree of the fitting polynomial

        Return:
            out (ndarray): smooth curve
        """
        # Convert input to ndarray
        data = self.make_input(data)
        
        N = len(data)
        # Generate x axis
        x = np.arange(1, N+1)
        # Fit a polynomial by least squares
        poly = np.polyfit(x, data, deg=poly_deg)
        # Evaluate the polynomial for all points to get a smooth curve
        out = np.poly1d(poly)(x)

        return out.astype(np.float32)