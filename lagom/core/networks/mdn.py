import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Normal
from torch.distributions.utils import log_sum_exp

from .mlp import MLP


class MDN(nn.Module):
    """
    This class defines the mixture density networks using isotropic Gaussian densities and MLP. 
    
    The network receives input tensor and outputs parameters for a mixture of Gaussian distributions. 
    i.e. mixing coefficients, means and variances. 
    
    Specifically, their dimensions are following, given N is batch size, K is the number of densities
    and D is the output dimension
    
    - mixing coefficients: [N, K, D]
    - mean: [N, K, D]
    - variance: [N, K, D]
    """
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 num_densities, 
                 hidden_sizes=None, 
                 hidden_nonlinearity=None):
        """
        Args:
            input_dim (int): the number of dimensions of input tensor
            output_dim (int): the number of dimensions of output tensor
            num_densities (int): the number of Gaussian densities for each output dimension
            hidden_sizes (list): a list of sizes for hidden layers
            hidden_nonlinearity (nn.functional): nonlinearity for hidden layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_densities = num_densities
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        
        # Create hidden layers if it is required
        if self.hidden_sizes is None:
            in_features = self.input_dim
        else:
            self.MLP = MLP(input_dim=self.input_dim, 
                           hidden_sizes=self.hidden_sizes, 
                           hidden_nonlinearity=self.hidden_nonlinearity, 
                           output_dim=None, 
                           output_nonlinearity=None)
            # Take out dimension of last hidden layer
            in_features = self.hidden_sizes[-1]
            
        # Create mixing coefficients head
        # Note that its unnormalized logits
        self.unnormalized_pi_head = nn.Linear(in_features=in_features, out_features=self.num_densities*self.output_dim)
        # Create mean head
        self.mu_head = nn.Linear(in_features=in_features, out_features=self.num_densities*self.output_dim)
        # Create log-variance head
        # Use log-variance allows to optimize values in [negative, 0, positive]
        # To retrieve std, use exp(2*log-variance) rather than sqrt for better numerical stability
        self.logvar_head = nn.Linear(in_features=in_features, out_features=self.num_densities*self.output_dim)
        
        # Initialize parameters
        self._init_params()
        
    def _init_params(self):
        """
        Initialize the network parameters, weights, biases
        
        Orthogonal weight initialization and zero bias initialization
        """
        # Initialization for mixing coefficients head
        # Weight initialization
        nn.init.orthogonal_(self.unnormalized_pi_head.weight, gain=1)  # gain=1 due to identity
        # Bias initialization
        nn.init.constant_(self.unnormalized_pi_head.bias, 0.0)
        
        # Initialization for mean head
        # Weight initialization
        nn.init.orthogonal_(self.mu_head.weight, gain=1)  # gain=1 due to identity
        # Bias initialization
        nn.init.constant_(self.mu_head.bias, 0.0)
        
        # Initialization for log-variance head
        # Weight initialization
        nn.init.orthogonal_(self.logvar_head.weight, gain=1)  # gain=1 due to identity
        # Bias initialization
        nn.init.constant_(self.logvar_head.bias, 0.0)
        
    def forward(self, x):
        # Enforce the shape of x to be consistent with first layer
        x = x.view(-1, self.input_dim)
        
        if self.hidden_sizes is not None:
            # Forward pass till last hidden layer via MLP
            x = self.MLP(x)
            
        # Forward pass through unnormalized mixing coefficients head
        unnormalized_pi = self.unnormalized_pi_head(x)
        # Convert to tensor with shape [N, K, D]
        unnormalized_pi = unnormalized_pi.view(-1, self.num_densities, self.output_dim)
        # Enforce each of coefficients are non-negative and summed up to 1
        # Note that it's LogSoftmax to compute numerically stable loss via log-sum-exp trick
        log_pi = F.log_softmax(unnormalized_pi, dim=1)
        
        # Forward pass through mean head
        mu = self.mu_head(x)
        # Convert mean to tensor with shape [N, K, D]
        mu = mu.view(-1, self.num_densities, self.output_dim)
        
        # Forward pass through log-variance head
        logvar = self.logvar_head(x)
        # Convert logvar to tensor with shape [N, K, D]
        logvar = logvar.view(-1, self.num_densities, self.output_dim)
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
        min_std = 1e-12
        
        log_probs = []
        
        # Iterate over all density components
        for i in range(self.num_densities):
            # Retrieve means and stds
            mu_i = mu[:, i, :]
            std_i = std[:, i, :]
            # Thresholding std, if std is 0, it leads to NaN loss. 
            #std_i = torch.clamp(std_i, min=min_std, max=std_i.max().item())
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
        target = target.view(-1, self.output_dim)
        
        # Calculate Gaussian log-probabilities over batch for each mixture and each data dimension
        log_gaussian_probs = self._calculate_batched_logprob(mu=mu, 
                                                             std=std, 
                                                             x=target)
        
        # Calculate the loss via log-sum-exp trick
        # Permute dimensions to [N, D, K] since log_sum_exp manipulate last dimension
        loss = -log_sum_exp(log_pi.permute(0, 2, 1) + log_gaussian_probs.permute(0, 2, 1))
        # Since log_sum_exp keepsdim, unsqueeze dimension to [N, D]
        loss = loss.squeeze(-1)
        
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