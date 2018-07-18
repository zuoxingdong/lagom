class BaseTestFunction(object):
    """
    Base class for test function in optimization.
    For more details, please refer to
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    
    All inherited subclasses should at least implement the following functions:
    1. __call__(self, x)
    """
    def __call__(self, x):
        raise NotImplementedError
