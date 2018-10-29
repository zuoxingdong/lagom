import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.distributions import Normal

from lagom.networks import BaseNetwork
from lagom.networks import make_fc
from lagom.networks import ortho_init


class MDN(BaseNetwork):
    def make_params(self, config):
        self.feature_layers = make_fc(1, [15, 15])
        
        self.pi_head = nn.Linear(15, config['num_density']*1)
        self.mean_head = nn.Linear(15, config['num_density']*1)
        self.logvar_head = nn.Linear(15, config['num_density']*1)
        
    def init_params(self, config):
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='tanh', constant_bias=0.0)
            
        ortho_init(self.pi_head, weight_scale=0.01, constant_bias=0.0)
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        ortho_init(self.logvar_head, weight_scale=0.01, constant_bias=0.0)
        
    def reset(self, config, **kwargs):
        pass
        
    def forward(self, x):
        for layer in self.feature_layers:
            x = torch.tanh(layer(x))
            
        # shape [N, K, D]
        logit_pi = self.pi_head(x).view(-1, self.config['num_density'], 1)
        mean = self.mean_head(x).view(-1, self.config['num_density'], 1)
        logvar = self.logvar_head(x).view(-1, self.config['num_density'], 1)
        std = torch.exp(0.5*logvar)
        
        return logit_pi, mean, std
    
    def mdn_loss(self, logit_pi, mean, std, target):
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
        
        dist = Categorical(pi)
        pi_samples = dist.sample()
        
        mean = mean[torch.arange(N*D), pi_samples]
        std = std[torch.arange(N*D), pi_samples]
        eps = torch.randn_like(std)
        samples = mean + eps*std*np.sqrt(tau)
        samples = samples.view(N, D)
        
        return samples
