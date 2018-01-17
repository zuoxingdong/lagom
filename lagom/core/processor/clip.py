import numpy as np

from lagom.core.processor import BaseProcessor


class Clip(BaseProcessor):
    def __init__(self, min_bound, max_bound):
        self.min_bound = min_bound
        self.max_bound = max_bound
        
    def process(self, x):
        """
        Clip values by min/max
        
        Args:
            x: input data
            
        Returns:
            out (numpy array): clipped data
        """
        
        x = self._make_input(x)
        
        out = np.clip(x, a_min=self.min_bound, a_max=self.max_bound)
        
        return out
        
    def _make_input(self, x):
        # make sure DType as numpy array
        if type(x) is not np.ndarray:
            x = np.array(x)
            
        return x