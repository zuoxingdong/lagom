class BaseES(object):
    """
    Base class for evolution strategies. 
    
    Note that the optimization should be treated as minimization. 
    
    All inherited subclasses should at least implement the following functions:
    1. ask(self)
    2. tell(self, solutions, function_values)
    3. @property: result(self)
    """
    def ask(self):
        """
        Sample new candidate solutions. 
        
        Returns:
            solutions (list/ndarray): sampled candidate solutions
        """
        raise NotImplementedError
        
    def tell(self, solutions, function_values):
        """
        Use the values of objective function evaluated for sampled solutions to prepare for next iteration.
        i.e. update the parameters of the population. 
        
        Args:
            solutions (list/ndarray): candidate solutions sampled from ask()
            function_values (list): objective function values evaluated for sampled solutions
        Returns:
        """
        raise NotImplementedError
        
    @property
    def result(self):
        """
        Return all necessary results after the optimization. 
        
        Returns:
            results (dict): a dictionary of results. 
                e.g. ['best_param', 'best_f_val', 'hist_best_param', 'stds']
        """
        raise NotImplementedError
