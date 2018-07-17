import numpy as np

from .base_test_function import BaseTestFunction


class HolderTable(BaseTestFunction):
    def __init__(self):
        self.search_domain = [-10, 10]
    
    def __call__(self, x):
        x, y = x
        
        y = -np.abs(np.sin(x)*np.cos(y)*np.exp(np.abs(1 - np.sqrt(x**2 + y**2)/np.pi)))
        
        return y
