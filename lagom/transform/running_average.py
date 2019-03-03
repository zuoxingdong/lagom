import numpy as np


class RunningAverage(object):
    r"""Keep a running average of a quantity. 
    
    Compared with estimating mean, it is more sentitive to recent changes. 
    
    Args:
        alpha (float): factor to control the sensitivity to recent changes, in the range [0, 1].
            Zero is most sensitive to recent change. 
    
    """
    def __init__(self, alpha):
        assert alpha >= 0 and alpha <= 1
        self.alpha = alpha
        
        self._value = None
        
    def __call__(self, x):
        r"""Update the estimate. 
        
        Args:
            x (object): additional data to update the estimation of running average. 
        """
        x = np.asarray(x, dtype=np.float32)
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha*self._value + (1 - self.alpha)*x
        return self.value
        
    @property
    def value(self):
        r"""Return the current running average. """
        return self._value
