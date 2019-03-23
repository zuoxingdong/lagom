import numpy as np


class PolyakAverage(object):
    r"""Keep a running average of a quantity via Polyak averaging. 
    
    Compared with estimating mean, it is more sentitive to recent changes. 
    
    Args:
        alpha (float): factor to control the sensitivity to recent changes, in the range [0, 1].
            Zero is most sensitive to recent change. 
    
    """
    def __init__(self, alpha):
        assert alpha >= 0 and alpha <= 1
        self.alpha = alpha
        
        self.x = None
        
    def __call__(self, x):
        r"""Update the estimate. 
        
        Args:
            x (object): additional data to update the estimation of running average. 
        """
        x = np.asarray(x, dtype=np.float32)
        if self.x is None:
            self.x = x
        else:
            self.x = self.alpha*self.x + (1 - self.alpha)*x
        return self.x
        
    def get_current(self):
        r"""Return the current running average. """
        return self.x
