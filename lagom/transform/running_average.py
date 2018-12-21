import numpy as np

from .base_transform import BaseTransform


class RunningAverage(BaseTransform):
    r"""Keep a running average of a quantity. 
    
    Compared with estimating mean, it is more sentitive to recent changes. 
    """
    def __init__(self, alpha):
        r"""Initialize the transform. 
        
        Args:
            alpha (float): factor to control the sensitivity to recent changes, in the range [0, 1].
                Zero is most sensitive to recent change. 
        """
        assert alpha >= 0 and alpha <= 1
        self.alpha = alpha
        
        self._value = None
        
    def __call__(self, x):
        r"""Update the estimate. 
        
        Args:
            x (object): additional data to update the estimation of running average. 
        """
        x = self.to_numpy(x, np.float32)
        
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha*self._value + (1 - self.alpha)*x
            
        return self.value
        
    @property
    def value(self):
        r"""Return the current running average. """
        return self._value
