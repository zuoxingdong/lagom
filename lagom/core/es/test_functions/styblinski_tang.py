from .base_test_function import BaseTestFunction


class StyblinskiTang(BaseTestFunction):
    def __init__(self):
        self.search_domain = [-5, 5]
    
    def __call__(self, x):
        y = 0.0
        for x_part in x:
            y += x_part**4 - 16*x_part**2 + 5*x_part

        return 0.5*y
