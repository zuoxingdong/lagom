class BaseAlgorithm(object):
    r"""Base class for all algorithms.
    
    Any algorithm should subclass this class. 
    
    The subclass should implement at least the following:
    
    - :meth:`__call__`
    """
    def __init__(self, name):
        r"""Initialize the algorithm.
        
        Args:
            name (str): name of the algorithm
        """
        self.name = name
        
    def __call__(self, config):
        r"""Run the algorithm with given configuration. 
        
        Args:
            config (OrderedDict): dictionary of configurations
            
        Returns:
            - result (object): result of the execution. If no need to return anything, 
                then an ``None`` should be returned. 
        """
        raise NotImplementedError
