class ESOptimizer(object):
    """
    Optimizer for evolution strategies (ES). 
    
    Each step it samples a solution set and evaluate 
    
    Note that this implementation is naive step-by-step version. For parallelized ES, please 
    refer to classes BaseESWorker and BaseESMaster. 
    """
    def __init__(self, es, f):
        """
        Args:
            es (BaseES): an instantiation of an evolution strategy algorithm. 
            f (object): objective function to compute fitness value
        """
        self.es = es
        self.f = f
        
    def step(self):
        """
        Perform one iteration (generation) of evolution strategy
        """
        # Sample candidate solutions from ES
        solutions = self.es.ask()
        # Compute objective function values of sampled candidate solutions
        function_values = [self.f(solution) for solution in solutions]
        # Update a new population
        self.es.tell(solutions, function_values)
        # Output results
        results = self.es.result
        
        return results
