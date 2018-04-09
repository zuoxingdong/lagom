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
        
    def run(self, config):
        """
        Run the algorithm with given configurations
        
        Args:
            config (OrderedDict): dictionary of configurations
        """
        raise NotImplementedError