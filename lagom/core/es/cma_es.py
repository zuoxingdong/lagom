from .base_es import BaseES


class CMAES(BaseES):
    """
    Wrapper of original CMA-ES implementation at https://github.com/CMA-ES/pycma
    
    Note that we minimize the objective, i.e. function values in tell(). 
    """
    def __init__(self, 
                 mu0, 
                 std0, 
                 popsize):
        """
        Args:
            mu0 (list or ndarray): initial mean
            std0 (float): initial standard deviation
            popsize (int): population size
        """
        self.mu0 = mu0
        self.std0 = std0
        self.popsize = popsize
        
        # Create CMA-ES instance
        import cma
        self.es = cma.CMAEvolutionStrategy(self.mu0, 
                                           self.std0, 
                                           {'popsize': self.popsize})
        
        self.solutions = None
        
    def ask(self):
        self.solutions = self.es.ask()
        return self.solutions
    
    def tell(self, solutions, function_values):
        # Update populations
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
