import numpy as np

from lagom.core.processor import BaseProcessor


class Standardize(BaseProcessor):
    def __init__(self, eps=np.finfo(np.float32).eps):
        self.eps = eps
    
    def process(self, x):
        """
        Standardize the input data: Subtracted by mean and divided by standard deviation
        
        Args:
            x (numpy array): input data.
            
        Returns:
            out (numpy array): standardize data
        """
        x = self._make_input(x)
        # Return clipped value if only one element, avoid zero output
        if x.shape[0] == 1:
            return np.clip(x, -1, 1)
        
        # Calculate mean and std for input vector
        mean = x.mean()
        std = x.std()
        
        out = (x - mean)/(std + self.eps)
        
        return out
    
    def _make_input(self, x):
        # make sure DType as numpy array
        if type(x) is not np.ndarray:
            x = np.array(x)
            
        # TODO: Currently only deal with single vector, maybe vectorization in Agent code ?
        # Add batch dimension if single vector
        #if len(x.shape) == 1:
        #    x = x.reshape([1, -1])
            
        return x