from abc import ABC
from abc import abstractmethod

from collections import namedtuple

import numpy as np

from lagom.transform import LinearSchedule


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
        
        Returns:
            list: a list of sampled candidate solutions
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
    def __init__(self, x0, sigma0, opts=None):
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

    
class CEM(BaseES):
    def __init__(self, 
                 x0, 
                 sigma0, 
                 opts=None):
        self.x0 = x0
        self.sigma0 = sigma0
        self.popsize = opts['popsize']
        self.elite_ratio = opts['elite_ratio']
        self.elite_size = max(1, int(self.elite_ratio*self.popsize))
        
        self.seed = opts['seed'] if 'seed' in opts else np.random.randint(1, 2**32)
        self.np_random = np.random.RandomState(self.seed)
        
        self.noise_scheduler = LinearSchedule(*opts['noise_scheduler_args'])
        self.iter = 0
        
        # initialize mean and std
        self.x = np.asarray(x0).astype(np.float32)
        self.shape = self.x.shape
        if np.isscalar(sigma0):
            self.sigma = np.full(self.shape, sigma0, dtype=np.float32)
        else:
            self.sigma = np.asarray(sigma0).astype(np.float32)
            
        self.xbest = None
        self.fbest = None

    def ask(self):
        extra_noise = self.noise_scheduler(self.iter)
        sigma = np.sqrt(self.sigma**2 + extra_noise)
        solutions = self.np_random.normal(self.x, sigma, size=(self.popsize,) + self.shape)
        return solutions
        
    def tell(self, solutions, function_values):
        solutions = np.asarray(solutions).astype(np.float32)
        elite_idx = np.argsort(function_values)[:self.elite_size]
        elite = solutions[elite_idx]
        
        self.x = elite.mean(axis=0)
        self.sigma = elite.std(axis=0)
        self.iter += 1
        
        self.xbest = elite[0]
        self.fbest = function_values[elite_idx[0]]
        
    @property
    def result(self):
        CEMResult = namedtuple('CEMResult', 
                               ['xbest', 'fbest', 'evals_best', 'evaluations', 'iterations', 'xfavorite', 'stds'],
                               defaults=[None]*7)
        result = CEMResult(xbest=self.xbest, fbest=self.fbest, iterations=self.iter, xfavorite=self.x, stds=self.sigma)
        return result
    
    def __repr__(self):
        return f'CEM in dimension {len(self.x0)} (seed={self.seed})'
