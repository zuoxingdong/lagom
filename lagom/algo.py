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
        
    def run(self, env, config, logger):
        """
        Run the algorithm with given environment and configurations
        
        Args:
            env (Env): environment object
            config (OrderedDict): dictionary of configurations
            logger (Logger): logger for current run of the algorithm
        """
        raise NotImplementedError