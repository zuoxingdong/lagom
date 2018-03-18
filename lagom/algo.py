class BaseAlgorithm(object):
    def run(self, env, config):
        """
        Run the algorithm with given environment and configurations
        
        Args:
            env (Env): environment object
            config (OrderedDict): dictionary of configurations
        """
        raise NotImplementedError