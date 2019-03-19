import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.networks import Module
from lagom.networks import ortho_init

from torch.distributions import Categorical
from torch.distributions import Normal


class MDNHead(Module):
    def __init__(self, in_features, out_features, num_density, device, **kwargs):
        super().__init__(**kwargs)
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_density = num_density
        self.device = device
        
        self.pi_head = nn.Linear(in_features, out_features*num_density)
        ortho_init(self.pi_head, weight_scale=0.01, constant_bias=0.0)
        self.mean_head = nn.Linear(in_features, out_features*num_density)
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        self.logvar_head = nn.Linear(in_features, out_features*num_density)
        ortho_init(self.logvar_head, weight_scale=0.01, constant_bias=0.0)
        
        self.to(self.device)
        
    def forward(self, x):
        logit_pi = self.pi_head(x).view(-1, self.num_density, self.out_features)
        mean = self.mean_head(x).view(-1, self.num_density, self.out_features)
        logvar = self.logvar_head(x).view(-1, self.num_density, self.out_features)
        std = torch.exp(0.5*logvar)
        return logit_pi, mean, std
        
    def loss(self, logit_pi, mean, std, target):
        r"""Calculate the MDN loss function. 
        
        The loss function (negative log-likelihood) is defined by:
        
        .. math::
            L = -\frac{1}{N}\sum_{n=1}^{N}\ln \left( \sum_{k=1}^{K}\prod_{d=1}^{D} \pi_{k}(x_{n, d})
            \mathcal{N}\left( \mu_k(x_{n, d}), \sigma_k(x_{n,d}) \right) \right)
            
        For better numerical stability, we could use log-scale:
        
        .. math::
            L = -\frac{1}{N}\sum_{n=1}^{N}\ln \left( \sum_{k=1}^{K}\exp \left\{ \sum_{d=1}^{D} 
            \ln\pi_{k}(x_{n, d}) + \ln\mathcal{N}\left( \mu_k(x_{n, d}), \sigma_k(x_{n,d}) 
            \right) \right\} \right) 
        
        .. note::
        
            One should always use the second formula via log-sum-exp trick. The first formula
            is numerically unstable resulting in +/- ``Inf`` and ``NaN`` error. 
        
        The log-sum-exp trick is defined by
        
        .. math::
            \log\sum_{i=1}^{N}\exp(x_i) = a + \log\sum_{i=1}^{N}\exp(x_i - a)
            
        where :math:`a = \max_i(x_i)`
        
        Args:
            logit_pi (Tensor): the logit of mixing coefficients, shape [N, K, D]
            mean (Tensor): mean of Gaussian mixtures, shape [N, K, D]
            std (Tensor): standard deviation of Gaussian mixtures, shape [N, K, D]
            target (Tensor): target tensor, shape [N, D]

        Returns
        -------
        loss : Tensor
            calculated loss
        """
        # target shape [N, D] to [N, 1, D]
        target = target.unsqueeze(1)
        
        log_pi = F.log_softmax(logit_pi, dim=1)
        
        dist = Normal(mean, std)
        log_probs = dist.log_prob(target)
        
        # [N, K, D] to [N, K]
        joint_log_probs = torch.sum(log_pi + log_probs, dim=-1, keepdim=False)
        # [N, K] to [N]
        loss = torch.logsumexp(joint_log_probs, dim=-1, keepdim=False)
        loss = -loss.mean(0)
        
        return loss
    
    def sample(self, logit_pi, mean, std, tau=1.0):
        r"""Sample from Gaussian mixtures using reparameterization trick.
        
        - Firstly sample categorically over mixing coefficients to determine a specific Gaussian
        - Then sample from selected Gaussian distribution
        
        Args:
            logit_pi (Tensor): the logit of mixing coefficients, shape [N, K, D]
            mean (Tensor): mean of Gaussian mixtures, shape [N, K, D]
            std (Tensor): standard deviation of Gaussian mixtures, shape [N, K, D]
            tau (float): temperature during sampling, it controls uncertainty. 
                * If :math:`\tau > 1`: increase uncertainty
                * If :math:`\tau < 1`: decrease uncertainty
        
        Returns
        -------
        x : Tensor
            sampled data with shape [N, D]
        """
        N, K, D = logit_pi.shape
        pi = F.softmax(logit_pi/tau, dim=1)
        # [N, K, D] to [N*D, K]
        pi = pi.permute(0, 2, 1).view(-1, K)
        mean = mean.permute(0, 2, 1).view(-1, K)
        std = std.permute(0, 2, 1).view(-1, K)
        
        pi_samples = Categorical(pi).sample()
        
        mean = mean[torch.arange(N*D), pi_samples]
        std = std[torch.arange(N*D), pi_samples]
        eps = torch.randn_like(std)
        samples = mean + eps*std*np.sqrt(tau)
        samples = samples.view(N, D)
        
        return samples
