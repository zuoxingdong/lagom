from abc import ABC
from abc import abstractmethod


class BaseES(ABC):
    r"""Base class for all evolution strategies. 
    
    .. note::
    
        The optimization is treated as minimization. e.g. maximize rewards is equivalent to minimize negative rewards.
        
    .. note::
    
        For painless parallelization, we highly recommend to use `concurrent.futures.ProcessPoolExecutor` with a few 
        practical tips. 
        
        * Set `max_workers` argument to control the max parallelization capacity. 
        * When execution get stuck, try to use :class:`CloudpickleWrapper` to wrap the objective function
          e.g. particularly for lambda, class methods
        * Use `with ProcessPoolExecutor` once to wrap entire iterative ES generations. Because using this 
          internally for each generation, it can slow down the parallelization dramatically due to overheads.
        * To reduce overheads further (e.g. PyTorch models, gym environments)
            * Recreate such models for each generation will be very expensive. 
            * Use initializer function for ProcessPoolExecutor
            * Within initializer function, define PyTorch models and gym environments as global variables
              Note that the global variables are defined to each worker independently
            * Don't forget to use `with torch.no_grad` to increase forward pass speed.

    """ 
    @abstractmethod
    def ask(self):
        r"""Sample a set of new candidate solutions. 
        
        Returns
        -------
        solutions : list
            sampled candidate solutions
        """
        pass
        
    @abstractmethod
    def tell(self, solutions, function_values):
        r"""Update the parameters of the population for a new generation based on the values of the objective
        function evaluated for sampled solutions. 
        
        Args:
            solutions (list/ndarray): candidate solutions returned from :meth:`ask`
            function_values (list): a list of objective function values evaluated for the sampled solutions.
        """
        pass
        
    @property
    @abstractmethod
    def result(self):
        r"""Return a namedtuple of all results for the optimization. 
        
        It contains:
        * xbest: best solution evaluated
        * fbest: objective function value of the best solution
        * evals_best: evaluation count when xbest was evaluated
        * evaluations: evaluations overall done
        * iterations: number of iterations
        * xfavorite: distribution mean in "phenotype" space, to be considered as current best estimate of the optimum
        * stds: effective standard deviations
        """
        pass

    
class CMAES(BaseES):
    r"""Implements CMA-ES algorithm. 
    
    .. note::
    
        It is a wrapper of the `original CMA-ES implementation`_. 
        
    Args:
        x0 (list): initial solution
        sigma0 (list): initial standard deviation
        opts (dict): a dictionary of options, e.g. ['popsize', 'seed']
        
    .. _original CMA-ES implementation:
        https://github.com/CMA-ES/pycma
    
    """
    def __init__(self, 
                 x0, 
                 sigma0, 
                 opts=None):
        r"""Initialize CMA-ES
        
        Args:
            mu0 (list/ndarray): initial mean
            std0 (float): initial standard deviation
            popsize (int): population size
        """
        import cma
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        
        self.x0 = self.es.x0
        self.sigma0 = self.es.sigma0
        self.popsize = self.es.popsize
        
    def ask(self):
        return self.es.ask()
    
    def tell(self, solutions, function_values):
        self.es.tell(solutions, function_values)
        
    @property
    def result(self):
        return self.es.result
