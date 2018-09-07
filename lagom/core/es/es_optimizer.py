class ESOptimizer(object):
    r"""A simple optimizer for evolution strategies (ES). 
    
    Each call of :meth:`step`, it samples a set of solution candidates, evaluate them
    to obtain the objective function values, do one ES update and return the results. 
    
    .. note::
    
        It is a naive step-by-step version for quite prototyping on toy problems. To parallelize ES, 
        see :class:`BaseESWorker` and :class:`BaseESMaster`. 
    
    """
    def __init__(self, es, f):
        r"""Initialize the ES optimizer. 
        
        Args:
            es (BaseES): an instantiation of an evolution strategy algorithm. 
            f (object): objective function to compute fitness value
        """
        self.es = es
        self.f = f
        
    def step(self):
        r"""Perform one iteration (generation) of evolution strategy. """
        # Sample candidate solutions from ES
        solutions = self.es.ask()
        # Compute objective function values of sampled candidate solutions
        function_values = [self.f(solution) for solution in solutions]
        # Update a new population
        self.es.tell(solutions, function_values)
        # Output results
        results = self.es.result
        
        return results
