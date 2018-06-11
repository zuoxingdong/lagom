import numpy as np

import torch
import torch.optim as optim

from lagom.core.preprocessors import Standardize



class BaseES(object):
    """
    Base class for evolution strategies
    """
    def ask(self):
        """
        Sample new candidate solutions. 
        
        Returns:
            solutions (numpy.ndarray): sampled candidate solutions
        """
        raise NotImplementedError
        
    def tell(self, solutions, function_values):
        """
        Use the values of objective function evaluated for sampled solutions to prepare for next iteration.
        i.e. update the parameters of the population. 
        
        Args:
            solutions (numpy.ndarray): candidate solutions sampled from ask()
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
        
    def l2_weight_decay(self, weight_decay, params):
        """
        L2 weight decay. 

        Args:
            weight_decay (float): coefficient for weight decay
            params (numpy.ndarray): the model parameters needed to have L2 weight decay. 
                Shape [N, D]: N is number of models, D is number of parameters for each model. 
        Returns:
            decayed_params (numpy.ndarray): model parameters after weight decay
        """
        assert params.ndim == 2

        decayed_params = params - weight_decay*np.mean(params**2, axis=-1, keepdims=True)

        return decayed_params
        

class CMAES(BaseES):
    """
    Wrapper of original CMA-ES implementation at https://github.com/CMA-ES/pycma
    """
    def __init__(self, 
                 mu0, 
                 std0, 
                 popsize, 
                 weight_decay=0.01):
        """
        Args:
            mu0 (list or ndarray): initial mean
            std0 (float): initial standard deviation
            popsize (int): population size
            weight_decay (float): weight decay
        """
        self.mu0 = mu0
        self.std0 = std0
        self.popsize = popsize
        self.weight_decay = weight_decay
        
        # Create CMA-ES instance
        import cma
        self.es = cma.CMAEvolutionStrategy(self.mu0, 
                                           self.std0, 
                                           {'popsize': self.popsize})
        
        self.solutions = None
        
    def ask(self):
        self.solutions = np.array(self.es.ask()) 
        return self.solutions
    
    def tell(self, solutions, function_values):
        # Weight decay if required
        if self.weight_decay > 0:
            solutions = self.l2_weight_decay(self.weight_decay, solutions)
        # Update populations
        self.es.tell(solutions, function_values)
        self.solutions = solutions
        
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
    
    
def compute_ranks(x):
    """
    Compute ranks of the input vector. The rank has the same dimension as input vector.
    Each element in the rank indicates the index of the ascendingly sorted input.
    i.e. ranks[i] = k, it means i-th element in the input is k-th smallest value. 
    
    Args:
        x (ndarray): Input array. Note that it should be only 1-dim vector
        
    Returns:
        ranks (ndarray): Ranks of input data
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    
    return ranks

def compute_centered_ranks(x):
    """
    Compute ranks such that their values are centered, [-0.5, 0.5]
    
    Args:
        x (ndarray): Input array. Note that it can be arbitrary shape
        
    Returns:
        centered_ranks (ndarray): centered ranks
    """
    
    ranks = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    
    centered_ranks = ranks/(x.size - 1) - 0.5
    
    return centered_ranks


class OpenAIES(BaseES):
    """
    Simple version of OpenAI evolution strategies.
    """
    def __init__(self, 
                 mu0, 
                 std0, 
                 popsize,
                 weight_decay=0.01, 
                 std_decay=0.999, 
                 min_std=0.01,
                 lr=1e-3, 
                 lr_decay=0.9999, 
                 min_lr=1e-2):
        """
        Args:
            mu0 (ndarray): initial mean
            std0 (float): initial standard deviation
            popsize (int): population size
            weight_decay (float): weight decay
            std_decay (float): standard deviation decay
            min_std (float): minimum of standard deviation
            lr (float): learning rate
            lr_decay (float): learning rate decay
            min_lr (float): minumum of learning rate
        """
        self.mu0 = np.array(mu0)
        self.std0 = std0
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.std_decay = std_decay
        self.min_std = min_std
        self.lr = lr
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        assert popsize % 2 == 0, 'popsize must be even, for antithetic sampling. '
        
        # Some other settings
        self.num_params = self.mu0.size
        self.mu = torch.from_numpy(self.mu0).float()
        self.mu.requires_grad = True  # requires gradient for optimizer to update
        self.std = self.std0
        self.optimizer = optim.Adam([self.mu], lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, 
                                                             gamma=self.lr_decay)
        
        self.solutions = None
        self.best_param = None
        self.best_f_val = None
        self.hist_best_param = None
        self.hist_best_f_val = None
    
    def ask(self):
        # Antithetic sampling of the standard Gaussian noise
        eps = np.random.randn(self.popsize//2, self.num_params)
        eps = np.concatenate([eps, -eps], axis=0)
        self.eps = eps  # record the noise for gradient computation in tell()
        
        # Perturbate the parameters
        self.solutions = self.mu.detach().numpy() + self.eps*self.std
        
        return self.solutions
        
    def tell(self, solutions, function_values):
        # Use centered ranks instead of raw values, avoid getting stuck in local optima in early stage
        centered_ranks = compute_centered_ranks(np.array(function_values))
        #centered_ranks = np.array(function_values)
        
        # Weight decay if required
        if self.weight_decay > 0:
            solutions = self.l2_weight_decay(self.weight_decay, solutions)
            
        # Make some results
        idx = np.argsort(function_values)[::-1][0]
        self.best_param = solutions[idx]
        self.best_f_val = function_values[idx]
        # Update the historical best result
        first_iteration = self.hist_best_param is None or self.hist_best_f_val is None
        if first_iteration or self.best_f_val < self.hist_best_f_val:
            self.hist_best_f_val = self.best_f_val
            self.hist_best_param = self.best_param
            
        # Compute gradient from original paper
        # Enforce fitness as Gaussian distributed, here we use centered ranks
        F = Standardize().process(centered_ranks)
        # Compute gradient, F:[popsize], eps: [popsize, num_params]
        grad = (1/self.std)*np.mean(np.expand_dims(F, 1)*self.eps, axis=0)
        grad = torch.from_numpy(grad).float()
        # Update the gradient to mu
        self.mu.grad = grad
        # Decay learning rate with lr scheduler
        self.lr_scheduler.step()
        # Take a gradient step
        self.optimizer.step()
        
        # Adaptive std
        if self.std > self.min_std:
            self.std = self.std_decay*self.std
        
    @property
    def result(self):
        results = {'best_param': self.best_param, 
                   'best_f_val': self.best_f_val, 
                   'hist_best_param': self.hist_best_param, 
                   'hist_best_f_val': self.hist_best_f_val,
                   'stds': self.std}
        
        return results
    
    
    
    
class ESOptimizer(object):
    def __init__(self, es_solver, f):
        """
        Args:
            es_solver (BaseES): evolution strategies
            f (object): objective function to compute fitness value
        """
        self.es_solver = es_solver
        self.f = f
        
    def step(self):
        """
        Perform one iteration (generation) of evolution strategy
        """
        f_vals = []
        # Sample candidate solutions from ES
        solutions = self.es_solver.ask()
        # Compute objective function values of sampled candidate solutions
        function_values = [self.f(solution) for solution in solutions]
        # Update a new population
        self.es_solver.tell(solutions, function_values)
        # Output results
        results = self.es_solver.result
        
        return results