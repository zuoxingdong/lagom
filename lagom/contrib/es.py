import numpy as np

import torch
import torch.optim as optim

from lagom.core.preprocessors import Standardize

from lagom.contrib.multiprocessing import BaseWorker, BaseMaster, BaseIterativeMaster


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
        self.solutions = np.array(self.es.ask()) 
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
    
    
def compute_ranks(x):
    """
    Rank transformation of the input vector. The rank has the same dimensionality as input vector.
    Each element in the rank indicates the index of the ascendingly sorted input.
    i.e. ranks[i] = k, it means i-th element in the input is k-th smallest value. 
    
    Rank transformation reduce sensitivity to outliers, e.g. in OpenAI ES, gradient computation
    involves fitness values in the population, if there are outliers (too large fitness), it affects
    the gradient too much. 
    
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
    
    Note that we minimize the objective, i.e. function values in tell(). 
    
    A practical tip, the learning rate is better to be proportional to batch size
    i.e. larger batch size, use larger learning rate and vise versa. 
    """
    def __init__(self, 
                 mu0, 
                 std0, 
                 popsize,
                 std_decay=0.999, 
                 min_std=0.01,
                 lr=1e-3, 
                 lr_decay=0.9999, 
                 min_lr=1e-2, 
                 antithetic=False,
                 rank_transform=True):
        """
        Args:
            mu0 (ndarray): initial mean
            std0 (float): initial standard deviation
            popsize (int): population size
            std_decay (float): standard deviation decay
            min_std (float): minimum of standard deviation
            lr (float): learning rate
            lr_decay (float): learning rate decay
            min_lr (float): minumum of learning rate
            antithetic (bool): If True, then use antithetic sampling to generate population.
            rank_transform (bool): If True, then use rank transformation of fitness (combat with outliers). 
        """
        self.mu0 = np.array(mu0)
        self.std0 = std0
        self.popsize = popsize
        self.std_decay = std_decay
        self.min_std = min_std
        self.lr = lr
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.antithetic = antithetic
        if self.antithetic:
            assert self.popsize % 2 == 0, 'popsize must be even for antithetic sampling. '
        self.rank_transform = rank_transform
        
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
        # Generate standard Gaussian noise for perturbating model parameters. 
        if self.antithetic:  # antithetic sampling
            eps = np.random.randn(self.popsize//2, self.num_params)
            eps = np.concatenate([eps, -eps], axis=0)
        else:
            eps = np.random.randn(self.popsize, self.num_params)
        # Record the noise for gradient computation in tell()
        self.eps = eps
        
        # Perturbate the parameters
        self.solutions = self.mu.detach().numpy() + self.eps*self.std
        
        return self.solutions
        
    def tell(self, solutions, function_values):
        # Enforce ndarray of function values
        function_values = np.array(function_values)
        if self.rank_transform:
            # Make a copy of original function values, for recording true values
            original_function_values = np.copy(function_values)
            # Use centered ranks instead of raw values, combat with outliers. 
            function_values = compute_centered_ranks(function_values)
            
        # Make some results
        # Sort function values and select the minimum, since we are minimizing the objective. 
        idx = np.argsort(function_values)[0]  # argsort is in ascending order
        self.best_param = solutions[idx]
        if self.rank_transform:  # use rank transform, we should record the original function values
            self.best_f_val = original_function_values[idx]
        else:
            self.best_f_val = function_values[idx]
        # Update the historical best result
        first_iteration = self.hist_best_param is None or self.hist_best_f_val is None
        if first_iteration or self.best_f_val < self.hist_best_f_val:
            self.hist_best_f_val = self.best_f_val
            self.hist_best_param = self.best_param
            
        # Compute gradient from original paper
        # Enforce fitness as Gaussian distributed, here we use centered ranks
        F = Standardize().process(function_values)
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
    
    
class ESWorker(BaseWorker):
    def __init__(self, f):
        """
        Args:
            f (function): objective function to evaluate the candidate.
        """
        self.f = f
    
    def work(self, master_cmd):
        # Unpack master command
        solution_id, solution, seed = master_cmd
        
        # Set random seed
        np.random.seed(seed)
        
        # Evaluate the solution to obtain fitness to the objective function
        function_value = self.f(solution)
        
        return solution_id, function_value
        
        
class ESMaster(BaseIterativeMaster):
    def __init__(self,
                 make_es,
                 num_iteration, 
                 worker, 
                 num_worker,
                 init_seed=0, 
                 daemonic_worker=None):
        super().__init__(num_iteration=num_iteration, 
                         worker=worker, 
                         num_worker=num_worker,
                         init_seed=init_seed, 
                         daemonic_worker=daemonic_worker)
        # Create ES solver
        self.es = make_es()
        # It is better to force popsize to be number of workers
        assert self.es.popsize == self.num_worker

    def make_tasks(self, iteration):
        # ES samples new candidate solutions
        solutions = self.es.ask()
        
        # Record iteration number, for logging in _process_workers_result()
        # And it also keeps API untouched for assign_tasks() in non-iterative Master class
        self.generation = iteration
        
        return solutions
        
    def _process_workers_result(self, tasks, workers_result):
        # Rename, in ES context, the task is to evalute the solution candidate
        solutions = tasks
        
        # Unpack function values from workers results, [solution_id, function_value]
        # Note that the workers result already sorted ascendingly with respect to task ID
        function_values = [result[1] for result in workers_result]
        
        # Update ES
        self.es.tell(solutions, function_values)
        
        # Obtain results from ES
        result = self.es.result
        
        # Unpack result
        best_f_val = result['best_f_val']
        if self.generation == 0 or (self.generation+1) % 100 == 0:
            print(f'Best function value at generation {self.generation+1}: {best_f_val}')
            
            
class GymESWorker(ESWorker):
    def work(self, master_cmd):
        # Unpack master
        solution_id, solution, seed = master_cmd
        
        # Unpack solution, since it contains [solution, env]
        solution, env = solution
        
        # Set random seed
        np.random.seed(seed)
        
        # Evaluate the solution with gym environment to obtain fitness 
        # fitness: negative rewards, since we are doing minimization in ES
        function_value = self.f(solution, env, seed)
        
        return solution_id, function_value
    
class GymESMaster(ESMaster):
    def __init__(self,
                 make_env,
                 make_es,
                 num_iteration, 
                 worker, 
                 num_worker,
                 init_seed=0, 
                 daemonic_worker=None):
        super().__init__(make_es=make_es, 
                         num_iteration=num_iteration, 
                         worker=worker, 
                         num_worker=num_worker,
                         init_seed=init_seed, 
                         daemonic_worker=daemonic_worker)
        
        # Create all gym environments on the top to save memory and overhead
        self.envs = [make_env() for _ in range(self.num_worker)]

    def make_tasks(self, iteration):
        # ES samples new candidate solutions
        solutions = self.es.ask()
        
        # Record iteration number, for logging in _process_workers_result()
        # And it also keeps API untouched for assign_tasks() in non-iterative Master class
        self.generation = iteration
        
        # Fit environment into solution together for worker to setup
        # e.g. worker can seed the environment and make rollout
        solutions = list(zip(solutions, self.envs))
        assert len(solutions) == self.es.popsize
        
        return solutions
        
    def _process_workers_result(self, tasks, workers_result):
        # Rename, in ES context, the task is to evalute the solution candidate
        # Unpack tasks, since it inclues [solutions, env]
        solutions = [task[0] for task in tasks]
        
        # Unpack function values from workers results, [solution_id, function_value]
        # Note that the workers result already sorted ascendingly with respect to task ID
        function_values = [result[1] for result in workers_result]
        
        # Update ES
        self.es.tell(solutions, function_values)
        
        # Obtain results from ES
        result = self.es.result
        
        # Unpack result
        best_f_val = result['best_f_val']
        if self.generation == 0 or (self.generation+1) % 100 == 0:
            print(f'Best function value at generation {self.generation+1}: {best_f_val}')