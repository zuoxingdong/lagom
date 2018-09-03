import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNetwork

####################
# TODO: update
# - do not overwrite __init__
# - put everything in make_params
# - in example, use make_fc/make_cnn


class BaseVAE(BaseNetwork):
    """
    Base class for variational autoencoders (VAE), works for both MLP and CNN versions.
    
    Note that if subclass overrides __init__, remember to provide
    keywords aguments, i.e. **kwargs passing to super().__init__. 
    
    All inherited subclasses should at least implement the following functions:
    1. make_encoder(self, config)
    2. make_moment_heads(self, config)
    3. make_decoder(self, config)
    4. init_params(self, config)
    5. encoder_forward(self, x)
    6. decoder_forward(self, x)
    
    Examples:
    
    class VAE(BaseVAE):
        def make_encoder(self, config):
            fc1 = nn.Linear(in_features=16, out_features=8)
            fc2 = nn.Linear(in_features=8, out_features=4)

            encoder = nn.ModuleList([fc1, fc2])

            return encoder

        def make_moment_heads(self, config):
            mu_head = nn.Linear(in_features=4, out_features=2)
            logvar_head = nn.Linear(in_features=4, out_features=2)

            return mu_head, logvar_head

        def make_decoder(self, config):
            fc1 = nn.Linear(in_features=2, out_features=4)
            fc2 = nn.Linear(in_features=4, out_features=8)
            fc3 = nn.Linear(in_features=8, out_features=16)

            decoder = nn.ModuleList([fc1, fc2, fc3])

            return decoder

        def init_params(self, config):
            gain = nn.init.calculate_gain('relu')

            for module in self.encoder:
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.constant_(module.bias, 0.0)

            nn.init.orthogonal_(self.mu_head.weight, gain=gain)
            nn.init.constant_(self.mu_head.bias, 0.0)
            nn.init.orthogonal_(self.logvar_head.weight, gain=gain)
            nn.init.constant_(self.logvar_head.bias, 0.0)

            for module in self.decoder:
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.constant_(module.bias, 0.0)

        def encoder_forward(self, x):
            for module in self.encoder:
                x = F.relu(module(x))

            return x

        def decoder_forward(self, x):
            for module in self.decoder[:-1]:
                x = F.relu(module(x))

            # Element-wise binary output
            x = torch.sigmoid(self.decoder[-1](x))

            return x
    """
    def __init__(self, config=None, **kwargs):
        # Override this constructor
        super(BaseNetwork, self).__init__()  # call nn.Module.__init__()
        
        self.config = config
        
        # Set all keyword arguments
        for key, val in kwargs.items():
            self.__setattr__(key, val)
        
        # Create encoder
        self.encoder = self.make_encoder(self.config)
        assert isinstance(self.encoder, nn.ModuleList)
        # Create heads for mean and log-variance (moments of latent variable)
        # Note that log operation allows to optimize negative values,
        # though variance must be non-negative
        self.mu_head, self.logvar_head = self.make_moment_heads(self.config)
        assert isinstance(self.mu_head, nn.Module)
        assert isinstance(self.logvar_head, nn.Module)
        # Create decoder
        self.decoder = self.make_decoder(self.config)
        assert isinstance(self.decoder, nn.ModuleList)
        
        # User-defined function to initialize all created parameters
        self.init_params(self.config)
        
    def make_params(self, config):
        # Not used, since we have additional user-defined functions for encoder/decoder
        pass
        
    def make_encoder(self, config):
        """
        User-defined function to create all the parameters (layers) for encoder
        
        Note that it must return a ModuleList, otherwise they cannot be tracked by PyTorch. 
        
        Args:
            config (Config): configurations
            
        Returns:
            encoder (ModuleList): ModuleList of encoder. 
            
        Examples:
            TODO
        """
        raise NotImplementedError
        
    def make_moment_heads(self, config):
        """
        User-defined function to create all the parameters (layers) for heads of mu and logvar. 
        
        Note that it must return a ModuleList, otherwise they cannot be tracked by PyTorch. 
        
        Args:
            config (Config): configurations
            
        Returns:
            mu_head (nn.Module): A module for mu head
            logvar_head (nn.Module): A module for logvar head
            
        Examples:
            TODO
        """
        raise NotImplementedError
        
    def make_decoder(self, config):
        """
        User-defined function to create all the parameters (layers) for decoder
        
        Note that it must return a ModuleList, otherwise they cannot be tracked by PyTorch. 
        
        Args:
            config (Config): configurations
            
        Returns:
            decoder (ModuleList): ModuleList of decoder. 
            
        Examples:
            TODO
        """
        raise NotImplementedError
        
    def encoder_forward(self, x):
        """
        User-defined function to define forward pass of encoder. 
        
        It should use the class member, self.encoder, 
        which is a ModuleList consisting of all defined parameters (layers) for encoder. 
        
        Args:
            x (Tensor): input tensor to encoder
            
        Returns:
            x (Tensor): features of encoder
            
        Examples:
            TODO
        """
        raise NotImplementedError
        
    def decoder_forward(self, x):
        """
        User-defined function to define forward pass of decoder. 
        
        It should use the class member, self.decoder, 
        which is a ModuleList consisting of all defined parameters (layers) for decoder. 
        
        Args:
            x (Tensor): the sampled latent variable according to output from moment heads
            
        Returns:
            x (Tensor): the reconstruction of the input
            
        Examples:
            TODO
        """
        raise NotImplementedError
        
    def reparameterize(self, mu, logvar):
        """
        Sampling using reparameterization trick
        
        i.e. mu + eps*std, eps sampled from N(0, 1)
        
        Args:
            mu (Tensor): mean of a Gaussian random variable
            logvar (Tensor): log-variance of a Gaussian random variable
                Note that log operation allows to optimize negative values,
                though variance must be non-negative.
        
        Returns:
            sampled tensor according to the reparameterization trick
        """
        # TODO: using PyTorch distributions rsample()
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
        x = self.encoder_forward(x)
        # Forward pass through moment heads to obtain mu and logvar for latent variable
        # Enforce features with shape [N, D], useful for ConvVAE
        x = x.view(x.size(0), -1)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        # Sample latent variable by using reparameterization trick
        z = self.reparameterize(mu, logvar)
        # Forward pass through decoder of sampled latent variable to obtain reconstructed input
        reconstructed_x = self.decoder_forward(z)
        
        return reconstructed_x, mu, logvar
    
    def calculate_loss(self, reconstructed_x, x, mu, logvar, reconstruction_loss_type='BCE'):
        """
        Calculate the VAE loss function
        VAE_loss = Reconstruction_loss + KL_loss
        Note that the losses are summed over all elements and batch
        
        For details, see https://arxiv.org/abs/1312.6114
        The KL loss is derived in Appendix B
        
        Args:
            reconstructed_x (Tensor): reconstructed x output from decoder
            x (Tensor): ground-truth x
            mu (Tensor): mean of the latent variable
            logvar (Tensor): log-variance of the latent variable
            loss_type (str): Type of reconstruction loss, supported ['BCE', 'MSE']
        
        Returns:
            loss (Tensor): VAE loss
        """
        # Reshape the reconstruction as [N, D]
        N = reconstructed_x.shape[0]
        reconstructed_x = reconstructed_x.view(N, -1)
        # Enforce the shape of x is the same as reconstructed x
        x = x.view_as(reconstructed_x)
        
        # Calculate reconstruction loss
        if reconstruction_loss_type == 'BCE':
            reconstruction_loss = F.binary_cross_entropy(reconstructed_x, 
                                                         x, 
                                                         reduction='none')  # all losses for [N, D]
        elif reconstruction_loss_type == 'MSE':
            reconstruction_loss = F.mse_loss(reconstructed_x, 
                                             x, 
                                             reduction='none')  # all losses for [N, D]
        # sum up loss for each data item, with shape [N]
        reconstruction_loss = reconstruction_loss.sum(1)
            
        # Calculate KL loss for each element
        # Gaussian: 0.5*sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KL_loss = -0.5*torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)
        
        # Compute total loss
        losses = reconstruction_loss + KL_loss
        
        # Average losses over batch
        loss = losses.mean()
        
        return loss, reconstruction_loss.mean(), KL_loss.mean()
