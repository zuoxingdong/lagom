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
        
    def __call__(self, config, seed):
        r"""Run the algorithm with a configuration and a random seed. 
        
        Args:
            config (dict): a dictionary of configuration items
            seed (int): a random seed to run the algorithm
            
        Returns
        -------
        result : object
            result of the execution. If no need to return anything, then an ``None`` should be returned. 
        """
        raise NotImplementedError
