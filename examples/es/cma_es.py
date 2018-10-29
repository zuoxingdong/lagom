from lagom.es import BaseES


class CMAES(BaseES):
    r"""Implements CMA-ES algorithm. 
    
    .. note::
    
        It is a wrapper of the `original CMA-ES implementation`_. 
        
    .. _original CMA-ES implementation:
        https://github.com/CMA-ES/pycma
    
    """
    def __init__(self, 
                 mu0, 
                 std0, 
                 popsize):
        r"""Initialize CMA-ES
        
        Args:
            mu0 (list/ndarray): initial mean
            std0 (float): initial standard deviation
            popsize (int): population size
        """
        self.mu0 = mu0
        self.std0 = std0
        self.popsize = popsize
        
        import cma
        self.es = cma.CMAEvolutionStrategy(self.mu0, 
                                           self.std0, 
                                           {'popsize': self.popsize})
        
    def ask(self):
        solutions = self.es.ask()
        return solutions
    
    def tell(self, solutions, function_values):
        self.es.tell(solutions, function_values)
        
    @property
    def result(self):
        # CMA-ES internal result
        # ['xbest', 'fbest', 'evals_best', 'evaluations', 'iterations', 'xfavorite', 'stds']
        results = self.es.result
        results = {'best_param': results[0],
                   'best_f_val': results[1], 
                   'hist_best_param': results[5], 
                   'stds': results[6]}
        
        return results
