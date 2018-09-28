from abc import ABC
from abc import abstractmethod

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.distributions import Normal

from .base_network import BaseNetwork


class BaseMDN(BaseNetwork, ABC):
    r"""Base class for Mixture Density Networks (we use Gaussian mixture). 
    
    This class defines the mixture density networks using isotropic Gaussian densities. 
    
    The network receives input tensor and outputs parameters for a mixture of Gaussian distributions. 
    i.e. mixing coefficients, means and variances. 
    
    Specifically, their dimensions are following, given N is batch size, K is the number of densities
    and D is the data dimension
    
    - mixing coefficients: ``[N, K, D]``
    - mean: ``[N, K, D]``
    - variance: ``[N, K, D]``
    
    The subclass should implement at least the following:

    - :meth:`make_feature_layers`
    - :meth:`make_mdn_heads`
    - :meth:`init_params`
    - :meth:`feature_forward`
    
    
    Example::
    
        class MDN(BaseMDN):
            def make_feature_layers(self, config):
                out = make_fc(input_dim=1, hidden_sizes=[15, 15])
                last_dim = 15

                return out, last_dim

            def make_mdn_heads(self, config, last_dim):
                out = {}

                num_density = 20
                data_dim = 1

                out['unnormalized_pi_head'] = nn.Linear(in_features=last_dim, out_features=num_density*data_dim)
                out['mu_head'] = nn.Linear(in_features=last_dim, out_features=num_density*data_dim)
                out['logvar_head'] = nn.Linear(in_features=last_dim, out_features=num_density*data_dim)
                out['num_density'] = num_density
                out['data_dim'] = data_dim

                return out

            def init_params(self, config):
                for layer in self.feature_layers:
                    ortho_init(layer, nonlinearity='tanh', weight_scale=1.0, constant_bias=0.0)

                ortho_init(self.unnormalized_pi_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)
                ortho_init(self.mu_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)
                ortho_init(self.logvar_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)

            def feature_forward(self, x):
                for layer in self.feature_layers:
                    x = torch.tanh(layer(x))

                return x
    
    """    
    def make_params(self, config):
        # Create feature layers
        self.feature_layers, self.last_dim = self.make_feature_layers(config)
        assert isinstance(self.feature_layers, nn.ModuleList)
        assert isinstance(self.last_dim, int)
        
        # Create MDN heads: unnormalized pi, mean and log-variance
        # also returns number of densities and data dimension
        out_heads = self.make_mdn_heads(self.config, last_dim=self.last_dim)
        assert isinstance(out_heads, dict) and len(out_heads) == 5
        # unpack
        self.unnormalized_pi_head = out_heads['unnormalized_pi_head']
        self.mu_head = out_heads['mu_head']
        self.logvar_head = out_heads['logvar_head']
        self.num_density = out_heads['num_density']
        self.data_dim = out_heads['data_dim']
        # sanity check
        assert isinstance(self.unnormalized_pi_head, nn.Module)
        assert isinstance(self.mu_head, nn.Module)
        assert isinstance(self.logvar_head, nn.Module)
        assert isinstance(self.num_density, int)
        assert isinstance(self.data_dim, int)
    
    @abstractmethod
    def make_feature_layers(self, config):
        r"""Create and return the parameters for all the feature layers. 
        
        .. note::
        
            For being able to track the parameters automatically, a ``nn.ModuleList`` should
            be returned. Also the dimension of last feature should also be returned.
            
        Args:
            config (dict): a dictionary of configurations. 
            
        Returns
        -------
        out : ModuleList
            a ModuleList of feature layers. 
        last_dim : int
            the dimension of last feature
        """
        pass

    @abstractmethod
    def make_mdn_heads(self, config, last_dim):
        r"""Create and returns all parameters/layers for MDN heads. 
        
        It includes the following:
        
        * ``unnormalized_pi_head pi``: a Module for mixing coefficient with output shape :math:`K\times D`
        * ``mu_head``: a Module for mean of Gaussian with output shape :math:`K\times D`
        * ``logvar_head``: a Module for log-variance of Gaussian with output shape :math:`K\times D`
        * ``num_density``: an integer :math:`K` number of densities
        * ``data_dim``: an integer :math:`D` dimension of data
        
        .. note::
        
            A dictionary of all created modules should be returned with the keys
            as their names. 
        
        Args:
            config (dict): a dictionary of configurations. 
            last_dim (int): last feature dimension helps to define layers for MDN heads. 
            
        Returns
        -------
        out : dict
            a dictionary of required output described above. 
        """
        pass
        
    @abstractmethod
    def feature_forward(self, x):
        r"""Defines forward pass of feature layers, before MDN heads. 
        
        .. note::
        
            It should use the class member ``self.feature_layers`` (a ModuleList). 
        
        Args:
            x (Tensor): input tensor
            
        Returns
        -------
        out : Tensor
            feature tensor before MDN heads
        """
        pass
        
    def forward(self, x):
        # Forward pass through feature layers to produce features before the MDN heads
        x = self.feature_forward(x)
        
        # Forward pass through the head of unnormalized pi (mixing coefficient)
        unnormalized_pi = self.unnormalized_pi_head(x)
        # Convert to tensor with shape [N, K, D]
        unnormalized_pi = unnormalized_pi.view(-1, self.num_density, self.data_dim)
        # Enforce each of coefficients are non-negative and summed up to 1
        # Note that it's LogSoftmax to compute numerically stable loss via log-sum-exp trick
        log_pi = F.log_softmax(unnormalized_pi, dim=1)
        
        # Forward pass through mean head
        mu = self.mu_head(x)
        # Convert to tensor with shape [N, K, D]
        mu = mu.view(-1, self.num_density, self.data_dim)
        
        # Forward pass through log-variance head
        logvar = self.logvar_head(x)
        # Convert to tensor with shape [N, K, D]
        logvar = logvar.view(-1, self.num_density, self.data_dim)
        # Retrieve std from logvar
        # For numerical stability: exp(0.5*logvar)
        # TODO: support softplus option, see `GaussianPolicy` class
        std = torch.exp(0.5*logvar)
        
        return log_pi, mu, std
    
    def calculate_batched_logprob(self, mu, std, x, _fast_code=True):
        r"""Calculate the log-probabilities for each data sampled by each density component. 
        Here the density is Gaussian. 
        
        .. warning::
        
            Currently there are fast and slow implementations temporarily with an option
            to select one to use. Once it is entirely sure the fast implementation is correct
            then this feature will be removed. A benchmark indicates that the fast implementation
            is roughly :math:`14x` faster !
        
        Args:
            mu (Tensor): mean of Gaussian mixtures, shape [N, K, D]
            std (Tensor): standard deviation of Gaussian mixtures, shape [N, K, D]
            x (Tensor): input tensor, shape [N, D]
            _fast_code (bool, optional): if ``True``, then using fast implementation. 
            
        Returns
        -------
        log_probs : Tensor
            the calculated log-probabilities for each data and each density, shape [N, K, D]
        """
        # Set up lower bound of std, since zero std can lead to NaN log-probability
        # Used for: torch.clamp(std_i, min=min_std...)
        # min_std = 1e-12
        
        def _fast(mu, std, x):
            # Create Gaussian distribution
            dist = Normal(loc=mu, scale=std)
            # Calculate the log-probabilities
            log_probs = dist.log_prob(x.unsqueeze(1).expand(-1, self.num_density, -1))
            
            return log_probs
        def _slow(mu, std, x):
            log_probs = []

            # Iterate over all density components
            for i in range(self.num_density):
                # Retrieve means and stds
                mu_i = mu[:, i, :]
                std_i = std[:, i, :]
                # Thresholding std, if std is 0, it leads to NaN loss. 
                # std_i = torch.clamp(std_i, min=min_std, max=std_i.max().item())
                # Create Gaussian distribution
                dist = Normal(loc=mu_i, scale=std_i)
                # Calculate the log-probability
                logp = dist.log_prob(x)
                # Record the log probability for current density
                log_probs.append(logp)

            # Stack log-probabilities with shape [N, K, D]
            log_probs = torch.stack(log_probs, dim=1)
            
            return log_probs
        
        # select code
        if _fast_code:
            return _fast(mu=mu, std=std, x=x)
        else:
            return _slow(mu=mu, std=std, x=x)

    def MDN_loss(self, log_pi, mu, std, target):
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
            log_pi (Tensor): log-scale mixing coefficients, shape [N, K, D]
            mu (Tensor): mean of Gaussian mixtures, shape [N, K, D]
            std (Tensor): standard deviation of Gaussian mixtures, shape [N, K, D]
            target (Tensor): target tensor, shape [N, D]

        Returns
        -------
        loss : Tensor
            calculated loss
        """
        # Enforce the shape of target to be consistent with output dimension
        target = target.view(-1, self.data_dim)
        
        # Calculate Gaussian log-probabilities over batch for each mixture and each data dimension
        log_gaussian_probs = self.calculate_batched_logprob(mu=mu, 
                                                            std=std, 
                                                            x=target, 
                                                            _fast_code=True)
        
        # Calculate the joint log-probabilities from [N, K, D] to [N, K]
        joint_log_probs = torch.sum(log_pi + log_gaussian_probs, dim=-1, keepdim=False)
        
        # Calculate the loss via log-sum-exp trick, from [N, K] to [N]
        # It calculates over K (mixing coefficient) dimension, produce tensor with shape [N]
        loss = -torch.logsumexp(joint_log_probs, dim=-1, keepdim=False)
        
        # Mean loss over the batch to scalar value
        loss = loss.mean(0)
        
        return loss
    
    def sample(self, log_pi, mu, std, tau=1.0, _fast_code=True):
        r"""Sample from Gaussian mixture using reparameterization trick.
        
        - Firstly sample categorically over mixing coefficients to determine a specific Gaussian
        - Then sample from selected Gaussian distribution
        
        .. warning::
        
            Currently there are fast and slow implementations temporarily with an option
            to select one to use. Once it is entirely sure the fast implementation is correct
            then this feature will be removed. A benchmark indicates that the fast implementation
            is roughly :math:`280x` faster on a large dataset !
        
        Args:
            log_pi (Tensor): log-scale mixing coefficients, shape [N, K, D]
            mu (Tensor): mean of Gaussian mixtures, shape [N, K, D]
            std (Tensor): standard deviation of Gaussian mixtures, shape [N, K, D]
            tau (float): temperature during sampling, it controls uncertainty. 
                * If :math:`\tau > 1`: increase uncertainty
                * If :math:`\tau < 1`: decrease uncertainty
            _fast_code (bool, optional): if ``True``, then using fast implementation. 
        
        Returns
        -------
        x : Tensor
            sampled data with shape [N, D]
        """
        # Get all shapes [batch size, number of densities, data dimension]
        N, K, D = log_pi.shape
        # Convert to [N*D, K], easy to use for Categorical probabilities
        log_pi = log_pi.permute(0, 2, 1).view(-1, K)
        mu = mu.permute(0, 2, 1).view(-1, K)
        std = std.permute(0, 2, 1).view(-1, K)
        
        # Get mixing coefficient
        if tau == 1.0:  # normal sampling, no uncertainty control
            pi = torch.exp(log_pi)
        else:  # use temperature
            pi = F.softmax(log_pi/tau, dim=1)  # now shape [N*D, K]
        # Create a categorical distribution for mixing coefficients
        pi_dist = Categorical(probs=pi)
        # Sampling mixing coefficients to determine which Gaussian to sample from for each data
        pi_samples = pi_dist.sample()  # shape [N*D]
        
        def _slow(mu, std, pi_samples):
            # Iteratively sample from selected Gaussian distributions
            samples = []
            for N_idx, pi_idx in enumerate(pi_samples):
                # Retrieve selected Gaussian distribution
                mu_i = mu[N_idx, pi_idx]
                std_i = std[N_idx, pi_idx]
                # Create standard Gaussian noise for reparameterization trick
                eps = torch.randn_like(std_i)
                # Sampling via reparameterization trick
                if tau == 1.0:  # normal sampling, no uncertainty control
                    samples.append(mu_i + eps*std_i)
                else:  # use temperature
                    samples.append(mu_i + eps*std_i*math.sqrt(tau))

            # Convert sampled data to a Tensor and reshape to [N, D]
            samples = torch.stack(samples, dim=0).view(N, D)

            return samples
        def _fast(mu, std, pi_samples):
            # Select mu and std with selected pi
            mu = mu[torch.arange(N*D), pi_samples.long()]
            std = std[torch.arange(N*D), pi_samples.long()]
            # Create standard Gaussian noise for reparameterization trick
            eps = torch.rand_like(std)
            # Sampling via reparameterization trick
            if tau == 1.0:  # normal sampling, no uncertainty control
                samples = mu + eps*std
            else:  # use temperature
                samples = mu + eps*std*math.sqrt(tau)
                
            # Reshape to [N, D]
            samples = samples.view(N, D)
                
            return samples
        
        # Select code
        if _fast_code:
            return _fast(mu=mu, std=std, pi_samples=pi_samples)
        else:
            return _slow(mu=mu, std=std, pi_samples=pi_samples)
