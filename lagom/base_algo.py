class BaseAlgorithm(object):
    """
    Base class for the algorithm. 
    
    All inherited subclasses should at least implement the following functions:
    1. __call__(self, config)
    """
    def __init__(self, name):
        """
        Args:
            name (str): name of the algorithm
        """
        self.name = name
        
    def __call__(self, config):
        """
        Run the algorithm with given configurations. 
        
        Args:
            config (OrderedDict): dictionary of configurations
            
        Returns:
            result (object): Result of the execution. 
                If no need to return anything, then `return None`
        """
        raise NotImplementedError
