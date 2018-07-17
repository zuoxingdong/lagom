import numpy as np

from .base_test_function import BaseTestFunction


class Rastrigin(BaseTestFunction):
    """
    Rastrigin test objective function for optimization
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    def __init__(self):
        self.search_domain = [-5.12, 5.12]
    
    def __call__(self, x):
        A = 10
        y = A*len(x)
        for x_part in x:
            y += x_part**2 - A*np.cos(2*np.pi*x_part)

        return y
