import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNetwork


class BaseVAE(BaseNetwork):
    r"""Base class for variational autoencoders (VAE). 
    
    This is a general class that could work for both MLP and CNN versions.
    
    The subclass should implement at least the following:
    
    - :meth:`make_encoder`
    - :meth:`make_moment_heads`
    - :meth:`make_decoder`
    - :meth:`init_params`
    - :meth:`encoder_forward`
    - :meth:`decoder_forward`
    
    Example::
    
    
    """
    def make_params(self, config):
        # Create encoder
        self.encoder, self.last_dim = self.make_encoder(config)
        assert isinstance(self.encoder, nn.ModuleList)
        assert isinstance(self.last_dim, int)
        
        # Create moment heads: mu and log-variance
        # also return the dimension for the latent variable z
        out_heads = self.make_moment_heads(config, last_dim=self.last_dim)
        assert isinstance(out_heads, dict) and len(out_heads) == 3
        # unpack
        self.mu_head = out_heads['mu_head']
        self.logvar_head = out_heads['logvar_head']
        self.z_dim = out_heads['z_dim']
        # sanity check
        assert isinstance(self.mu_head, nn.Module)
        assert isinstance(self.logvar_head, nn.Module)
        assert isinstance(self.z_dim, int)
        
        # Create decoder
        self.decoder = self.make_decoder(config, z_dim=self.z_dim)
        assert isinstance(self.decoder, nn.ModuleList)
        
    def make_encoder(self, config):
        r"""Create and return all the parameters/layers for the encoder. 
        
        .. note::
        
            For being able to track the parameters automatically, a ``nn.ModuleList`` should
            be returned. Also the dimension of last feature should also be returned.
        
        Args:
            config (dict): a dictionary of configurations. 
            
        Returns
        -------
        out : ModuleList
            a ModuleList of encoder
        last_dim : int
            the dimension of last feature
        """
        raise NotImplementedError
        
    def make_moment_heads(self, config, last_dim):
        r"""Create and return all the parameters for mu and logvar heads. 
        
        It includes the following:
        
        * ``mu_head``: a Module for mean of latent Gaussian.
        * ``logvar_head``: a Module for log-variance of latent Gaussian. 
        * ``z_dim``: an integer of the latent variable dimension. 
        
        .. note::
        
            A dictionary of all created modules should be returned with the keys
            as their names. 
        
        Args:
            config (dict): a dictionary of configurations. 
            z_dim (int): the dimension of latent variable
            
        Returns
        -------
        out : dict
            a dictionary of required output described above. 
        """
        raise NotImplementedError
        
    def make_decoder(self, config, z_dim):
        r"""Create and return all the parameters/layers for the decoder. 
        
        .. note::
        
            For being able to track the parameters automatically, a ``nn.ModuleList`` should
            be returned.
        
        Args:
            config (dict): a dictionary of configurations. 
            z_dim (int): the dimension of latent variable
            
        Returns
        -------
        out : ModuleList
            a ModuleList of decoder
        """
        raise NotImplementedError
        
    def encoder_forward(self, x):
        r"""Defines forward pass of encoder. 
        
        .. note::
        
            It should use the class member ``self.encoder`` (a ModuleList). 
        
        Args:
            x (Tensor): input tensor
            
        Returns
        -------
        out : Tensor
            feature tensor before moment heads of latent variable
        """
        raise NotImplementedError
        
    def decoder_forward(self, z):
        r"""Defines forward pass of decoder. 
        
        .. note::
        
            It should use the class member ``self.decoder`` (a ModuleList)
        
        Args:
            z (Tensor): the sampled latent variable from moment heads
            
        Returns
        -------
        re_x : Tensor
            the reconstruction of the input
        """
        raise NotImplementedError
        
    def reparameterize(self, mu, logvar):
        r"""Sampling using reparameterization trick. 
        
        .. note::
        
            It is a differentiable transformation, so one could use backpropagation
            through it. 
            
        Formally, it does :math:`\mu + \epsilon\cdot\sigma` where :math:`\epsilon\sim\mathcal{N}(0, 1)`
        
        Args:
            mu (Tensor): mean of a Gaussian random variable
            logvar (Tensor): log-variance of a Gaussian random variable
                Note that log operation allows to optimize negative values, though 
                variance must be non-negative.
        
        Returns
        -------
        out : Tensor
            sampled tensor according to the reparameterization trick
        """
        # TODO: add option for std parameterization (softplus), see `GaussianPolicy`
        if self.training:  # training: sample with reparameterization trick
            # Recover std from log-variance
            # 0.5*logvar by logarithm law is more numerically stable than taking square root
            std = torch.exp(0.5*logvar)
            # Sample standard Gaussian noise
            eps = torch.randn_like(std)
            
            return mu + eps*std
        else:  # evaluation: no sampling, simply pass mu
            return mu
        
    def forward(self, x):
        # Forward pass through encoder to obtain features
        features = self.encoder_forward(x)
        
        # Flatten features with shape [N, D], useful for ConvVAE
        features = features.flatten(start_dim=1)
        
        # Forward pass through moment heads to obtain mu and logvar for latent variable
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        
        # Sample latent variable by using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Forward pass through decoder of sampled latent variable to obtain reconstructed input
        re_x = self.decoder_forward(z)
        
        return re_x, mu, logvar
    
    def calculate_loss(self, re_x, x, mu, logvar, loss_type='BCE'):
        r"""Calculate `VAE loss function`_. 
        
        The VAE loss is the summation of reconstruction loss and KL loss. 
        
        .. _VAE loss function:
            https://arxiv.org/abs/1312.6114
        
        .. note::
        
            The losses are summed over all elements and batch. 
        
        Args:
            re_x (Tensor): reconstructed input returned from decoder
            x (Tensor): ground-truth input
            mu (Tensor): mean of the latent variable
            logvar (Tensor): log-variance of the latent variable
            loss_type (str): Type of reconstruction loss, supported ['BCE', 'MSE']
        
        Returns
        -------
        out : dict
            a dictionary of selected output such as loss, reconstruction loss and KL loss. 
        """
        assert loss_type in ['BCE', 'MSE'], f'expected either BCE or MSE, got {loss_type}'
        
        out = {}
        
        # flatten the reconstructed input as [N, D]
        re_x = re_x.flatten(start_dim=1)
        
        # Enforce same shape of x as reconstructed x
        x = x.view_as(re_x)
        
        # make loss function
        if loss_type == 'BCE':
            loss_f = F.binary_cross_entropy
        elif loss_type == 'MSE':
            loss_f = F.mse_loss
        else:
            raise ValueError
            
        # Calculate reconstruction loss for all elements in [N, D]
        re_loss = loss_f(input=re_x, target=x, reduction='none')
        # Sum up over data dimension, from [N, D] to [N]
        re_loss = re_loss.sum(1)
        
        # Calculate KL loss for each element to shape [N]
        # Gaussian: -0.5*sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KL_loss = -0.5*torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)
        
        # Compute total loss with shape [N]
        losses = re_loss + KL_loss
        
        # Record output
        # Average losses over batch
        out['loss'] = losses.mean()
        out['re_loss'] = re_loss.mean()
        out['KL_loss'] = KL_loss.mean()
        
        return out
