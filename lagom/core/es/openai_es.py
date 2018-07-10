import numpy as np

import torch
import torch.optim as optim

from .base_es import BaseES

from lagom.core.transform import Standardize
from lagom.core.transform import RankTransform


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
        if self.rank_transform:
            self.rank_transformer = RankTransform()
        
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
        
        return list(self.solutions)
        
    def tell(self, solutions, function_values):
        # Enforce ndarray of function values
        function_values = np.array(function_values)
        if self.rank_transform:
            # Make a copy of original function values, for recording true values
            original_function_values = np.copy(function_values)
            # Use centered ranks instead of raw values, combat with outliers. 
            function_values = self.rank_transformer(function_values, centered=True)
            
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
        standardize = Standardize()
        F = standardize(function_values)
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