from collections import namedtuple

import numpy as np

import torch
import torch.optim as optim

from lagom import BaseES
from lagom.transform import rank_transform
from lagom.transform import LinearSchedule


class OpenAIES(BaseES):
    r"""Implements OpenAI evolution strategies.
    
    .. note::
    
        In practice, the learning rate is better to be proportional to the batch size.
        i.e. for larger batch size, use larger learning rate and vise versa. 
        
    Args:
        x0 (ndarray): initial mean
        sigma0 (float): initial standard deviation
        popsize (int): population size
        sigma_scheduler_args (list): arguments for linear scheduling of standard deviation
        lr (float): learning rate
        lr_decay (float): learning rate decay
        min_lr (float): minumum of learning rate
        antithetic (bool): If True, then use antithetic sampling to generate population.
        rank_transform (bool): If True, then use rank transformation of fitness (combat with outliers). 
        
    """
    def __init__(self, x0, sigma0, opts=None):
        self.x0 = x0
        self.sigma0 = sigma0
        self.popsize = opts['popsize']
        self.sigma_scheduler = LinearSchedule(*opts['sigma_scheduler_args'])
        self.lr = opts['lr']
        self.lr_decay = opts['lr_decay']
        self.min_lr = opts['min_lr']
        self.antithetic = opts['antithetic']
        if self.antithetic:
            assert self.popsize % 2 == 0, 'popsize must be even for antithetic sampling. '
        self.rank_transform = opts['rank_transform']
        
        self.seed = opts['seed'] if 'seed' in opts else np.random.randint(1, 2**32)
        self.np_random = np.random.RandomState(self.seed)
        
        self.iter = 0

        # initialize mean and std
        self.x = torch.from_numpy(np.asarray(x0)).float()
        self.x.requires_grad = True
        self.shape = tuple(self.x.shape)
        if np.isscalar(sigma0):
            self.sigma = np.full(self.shape, sigma0, dtype=np.float32)
        else:
            self.sigma = np.asarray(sigma0).astype(np.float32)
        
        self.optimizer = optim.Adam([self.x], lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        
        self.xbest = None
        self.fbest = None

    def ask(self):
        if self.antithetic:
            eps = self.np_random.randn(self.popsize//2, *self.shape)
            eps = np.concatenate([eps, -eps], axis=0)
        else:
            eps = self.np_random.randn(self.popsize, *self.shape)
        self.eps = eps
        
        solutions = self.x.detach().cpu().numpy() + self.eps*self.sigma
        return solutions
        
    def tell(self, solutions, function_values):
        solutions = np.asarray(solutions).astype(np.float32)
        function_values = np.asarray(function_values).astype(np.float32)
        if self.rank_transform:
            original_function_values = np.copy(function_values)
            function_values = rank_transform(function_values, centered=True)  # center: combat outliers
        idx = np.argsort(function_values)
        self.xbest = solutions[idx[0]]
        if self.rank_transform:  # record original function values
            self.fbest = original_function_values[idx[0]]
        else:
            self.fbest = function_values[idx[0]]
        
        # Enforce fitness as Gaussian distributed, also for centered ranks
        F = (function_values - function_values.mean(-1))/(function_values.std(-1) + 1e-8)
        # Compute gradient, F:[popsize], eps: [popsize, num_params]
        self.optimizer.zero_grad()
        grad = (1/self.sigma)*np.mean(np.expand_dims(F, 1)*self.eps, axis=0)
        grad = torch.from_numpy(grad).float()
        self.x.grad = grad
        self.lr_scheduler.step()
        self.optimizer.step()
        
        self.iter += 1
        self.sigma = self.sigma_scheduler(self.iter)
        
    @property
    def result(self):
        OpenAIESResult = namedtuple('OpenAIESResult', 
                                    ['xbest', 'fbest', 'evals_best', 'evaluations', 'iterations', 'xfavorite', 'stds'],
                                    defaults=[None]*7)
        result = OpenAIESResult(xbest=self.xbest, fbest=self.fbest, iterations=self.iter, xfavorite=self.x.detach().cpu().numpy(), stds=self.sigma)
        return result
        
    def __repr__(self):
        return f'OpenAIES in dimension {len(self.x0)} (seed={self.seed})'
