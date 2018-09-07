class BaseTestFunction(object):
    r"""Base class for all test functions in optimization.
    
    For more details, please refer to `this Wikipedia page`_. 
    
    .. _this Wikipedia page:
        https://en.wikipedia.org/wiki/Test_functions_for_optimization
    
    The subclass should implement at least the following:
    
    - :meth:`__call__`
    
    """
    def __call__(self, x):
        raise NotImplementedError
