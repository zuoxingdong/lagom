import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Normal

from .base_network import BaseNetwork

####################
# TODO: update
# - do not overwrite __init__
# - put everything in make_params
# - in example, use make_fc/make_cnn

class BaseMDN(BaseNetwork):
    """
    Base class for mixture density networks (we use Gaussian mixture). 
    
    This class defines the mixture density networks using isotropic Gaussian densities. 
    
    The network receives input tensor and outputs parameters for a mixture of Gaussian distributions. 
    i.e. mixing coefficients, means and variances. 
    
    Specifically, their dimensions are following, given N is batch size, K is the number of densities
    and D is the data dimension
    
    - mixing coefficients: [N, K, D]
    - mean: [N, K, D]
    - variance: [N, K, D]
    
    Note that if subclass overrides __init__, remember to provide
    keywords aguments, i.e. **kwargs passing to super().__init__. 
    
    All inherited subclasses should at least implement the following functions:
    1. make_feature_layers(self, config)
    2. make_mdn_heads(self, config)
    3. init_params(self, config)
    4. feature_forward(self, x)
    
    Examples:
    
    class MDN(BaseMDN):
        def make_feature_layers(self, config):
            fc1 = nn.Linear(in_features=1, out_features=15)
            fc2 = nn.Linear(in_features=15, out_features=15)

            feature_layers = nn.ModuleList([fc1, fc2])

            return feature_layers

        def make_mdn_heads(self, config):
            unnormalized_pi_head = nn.Linear(in_features=15, out_features=20*1)
            mu_head = nn.Linear(in_features=15, out_features=20*1)
            logvar_head = nn.Linear(in_features=15, out_features=20*1)

            num_densities = 20
            data_dim = 1

            return unnormalized_pi_head, mu_head, logvar_head, num_densities, data_dim

        def init_params(self, config):
            for module in self.feature_layers:
                gain = nn.init.calculate_gain('tanh')

                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.constant_(module.bias, 0.0)

            nn.init.orthogonal_(self.unnormalized_pi_head.weight, gain=gain)
            nn.init.constant_(self.unnormalized_pi_head.bias, 0.0)

            nn.init.orthogonal_(self.mu_head.weight, gain=gain)
            nn.init.constant_(self.mu_head.bias, 0.0)

            nn.init.orthogonal_(self.logvar_head.weight, gain=gain)
            nn.init.constant_(self.logvar_head.bias, 0.0)

        def feature_forward(self, x):
            for module in self.feature_layers:
                x = torch.tanh(module(x))

            return x
    """
    def __init__(self, config=None, **kwargs):
        # Override this constructor
        super(BaseNetwork, self).__init__()  # call nn.Module.__init__()
        
        self.config = config
        
        # Set all keyword arguments
        for key, val in kwargs.items():
            self.__setattr__(key, val)
        
        # Create feature layers
        self.feature_layers = self.make_feature_layers(self.config)
        assert isinstance(self.feature_layers, nn.ModuleList)
        # Create MDN heads for unnormalized pi (mixing coeffcients), mean and log-variance
        # Note that log operation allows to optimize including negative values,
        # though variance must be non-negative
        out_heads = self.make_mdn_heads(self.config)
        self.unnormalized_pi_head, self.mu_head, self.logvar_head = out_heads[:3]
        self.num_densities, self.data_dim = out_heads[3:]
        assert isinstance(self.unnormalized_pi_head, nn.Module)
        assert isinstance(self.mu_head, nn.Module)
        assert isinstance(self.logvar_head, nn.Module)
        assert isinstance(self.num_densities, int)
        assert isinstance(self.data_dim, int)
        
        # User-defined function to initialize all created parameters
        self.init_params(self.config)
        
    def make_params(self, config):
        # Not used, since we have additional user-defined functions for encoder/decoder
        pass
    
    def make_feature_layers(self, config):
        """
        User-defined function to create the parameters for all the feature layers. 
        
        Note that it must return a ModuleList, otherwise they cannot be tracked by PyTorch. 
        
        Args:
            config (Config): configurations
            
        Returns:
            feature_layers (nn.ModuleList): ModuleList of feature layers
            
        Examples:
            TODO:
        """
        raise NotImplementedError
        
    def make_mdn_heads(self, config):
        """
        User-defined function to create all the parameters (layers) for heads of
        unnormalized pi (mixing coefficient), mu and logvar. Note that they should
        output dimensions, K*D, where K is the number of densities and D is 
        the data dimensions. 
        
        Note that it must return a ModuleList, otherwise they cannot be tracked by PyTorch. 
        
        Args:
            config (Config): configurations
            
        Returns:
            unnormalized_pi_head (nn.Module): A module for un-normalized pi (mixing coefficients). 
            mu_head (nn.Module): A module for mean head. 
            logvar_head (nn.Module): A module for log-variance head. 
            num_densities (int): number of densities
            data_dim (int): data dimension
            
        Examples:
            TODO
        """
        raise NotImplementedError
        
    def feature_forward(self, x):
        """
        User-defined function to define forward pass of feature layers, before MDN heads. 
        
        It should use the class member, self.feature_layers, 
        which is a ModuleList consisting of all defined parameters (layers). 
        
        Args:
            x (Tensor): input tensor
            
        Returns:
            x (Tensor): feature tensor before the MDN heads. 
            
        Examples:
            TODO
        """
        raise NotImplementedError
        
    def forward(self, x):
        # Forward pass through feature layers to produce features before the MDN heads
        x = self.feature_forward(x)
        
        # Forward pass through the head of unnormalized pi (mixing coefficient)
        unnormalized_pi = self.unnormalized_pi_head(x)
        # Convert to tensor with shape [N, K, D]
        unnormalized_pi = unnormalized_pi.view(-1, self.num_densities, self.data_dim)
        # Enforce each of coefficients are non-negative and summed up to 1
        # Note that it's LogSoftmax to compute numerically stable loss via log-sum-exp trick
        log_pi = F.log_softmax(unnormalized_pi, dim=1)
        
        # Forward pass through mean head
        mu = self.mu_head(x)
        # Convert to tensor with shape [N, K, D]
        mu = mu.view(-1, self.num_densities, self.data_dim)
        
        # Forward pass through log-variance head
        logvar = self.logvar_head(x)
        # Convert to tensor with shape [N, K, D]
        logvar = logvar.view(-1, self.num_densities, self.data_dim)
        # Retrieve std from logvar
        # For numerical stability: exp(0.5*logvar)
        std = torch.exp(0.5*logvar)
        
        return log_pi, mu, std
    
    def _calculate_batched_logprob(self, mu, std, x):
        """
        Calculate the log-probabilities for each data sampled by each density component. 
        Here the density is Gaussian. 
        
        Args:
            mu (Tensor): mean of Gaussian mixtures, shape [N, K, D]
            std (Tensor): standard deviation of Gaussian mixtures, shape [N, K, D]
            x (Tensor): input tensor, shape [N, D]
            
        Returns:
            log_probs (Tensor): the calculated log-probabilities for each data and each density, shape [N, K, D]
        """
        # Set up lower bound of std, since zero std can lead to NaN log-probability
        # Used for: torch.clamp(std_i, min=min_std...)
        # min_std = 1e-12
        
        log_probs = []
        
        # Iterate over all density components
        for i in range(self.num_densities):
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
    
    def MDN_loss(self, log_pi, mu, std, target):
        """
        Calculate the loss function
        
        i.e. negative log-likelihood of the target given the parameters of Gaussian mixtures
        L = -\frac{1}{N}\sum_{n=1}^{N}(\ln(\sum_{k=1}^{K} pi_k*Gaussian probability))
        For better numerical stability, we could use log-scale of denstiy mixture 
        L = -\frac{1}{N}\sum_{n=1}^{N}(\ln(\sum_{k=1}^{K} \exp( \log pi_k + \log Gaussian probability)) )
        
        Note that simply computing this loss function is numerically unstable.
        Due to the fact that the density mixture might be very small value, resulting in +/- Inf.
        To address this problem, we use log-sum-exp trick
        
        i.e. \log\sum_{i=1}^{N}\exp(x_i) = a + \log\sum_{i=1}^{N}\exp(x_i - a), where a = \max_i(x_i)
        
        Args:
            log_pi (Tensor): log-scale mixing coefficients, shape [N, K, D]
            mu (Tensor): mean of Gaussian mixtures, shape [N, K, D]
            std (Tensor): standard deviation of Gaussian mixtures, shape [N, K, D]
            target (Tensor): target tensor, shape [N, D]

        Returns:
            loss (Tensor): calculated loss
        """
        # Enforce the shape of target to be consistent with output dimension
        target = target.view(-1, self.data_dim)
        
        # Calculate Gaussian log-probabilities over batch for each mixture and each data dimension
        log_gaussian_probs = self._calculate_batched_logprob(mu=mu, 
                                                             std=std, 
                                                             x=target)
        
        # Calculate the loss via log-sum-exp trick
        # It calculates over K (mixing coefficient) dimension, produce tensor with shape [N, D]
        loss = -torch.logsumexp(log_pi + log_gaussian_probs, dim=1, keepdim=False)
        
        # Sum up loss over elements and average over batch
        loss = loss.sum(1).mean()
        
        return loss
    
    def sample(self, log_pi, mu, std, tau=1.0):
        """
        Sampling from Gaussian mixture using reparameterization trick.
        
        - Firstly sample categorically from mixing coefficients to determine a Gaussian distribution
        - Then sample from selected Gaussian distribution
        
        Args:
            log_pi (Tensor): log-scale mixing coefficients, shape [N, K, D]
            mu (Tensor): mean of Gaussian mixtures, shape [N, K, D]
            std (Tensor): standard deviation of Gaussian mixtures, shape [N, K, D]
            tau (float): temperature during sampling, controlling uncertainty. 
                If tau > 1: increase uncertainty
                If tau < 1: decrease uncertainty
        
        Returns:
            x (Tensor): sampled data, shape [N, D]
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
            pi = F.softmax(log_pi/tau, dim=1)
        # Create a categorical distribution for mixing coefficients
        pi_dist = Categorical(probs=pi)
        # Sampling mixing coefficients to determine which Gaussian to sample from for each data
        pi_samples = pi_dist.sample()
        # Convert 
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
