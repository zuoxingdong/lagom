class BaseAlgorithm(object):
    """
    Base class for the algorithm
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