from .base_test_function import BaseTestFunction


class Sphere(BaseTestFunction):
    def __init__(self, min=-1000, max=1000):
        self.search_domain = [min, max]
    
    def __call__(self, x):
        y = 0.0
        for x_part in x:
            y += x_part**2

        return y