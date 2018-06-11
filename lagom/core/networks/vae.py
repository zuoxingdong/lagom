import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class VAE(nn.Module):
    """
    Variational Autoencoders (VAE) with MLP
    """
    def __init__(self, 
                 input_dim, 
                 encoder_sizes, 
                 encoder_nonlinearity, 
                 latent_dim, 
                 decoder_sizes, 
                 decoder_nonlinearity, 
                 decoder_output_nonlinearity):
        """
        Set up VAE with configurations
        
        Args:
            input_dim (int): input dimension
            encoder_sizes (list): a list of sizes for encoder hidden layers
            encoder_nonlinearity (nn.functional): nonlinearity for encoder hidden layers
            latent_dim (int): latent dimension
            decoder_sizes (list): a list of sizes for decoder hidden layers
            decoder_nonlinearity (nn.functional): nonlinearity for decoder hidden layers
            decoder_output_nonlinearity (nn.functional): nonlinearity 
                for final decoder transposed convolutional layer. e.g. sigmoid for BCE loss
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.encoder_sizes = encoder_sizes
        self.encoder_nonlinearity = encoder_nonlinearity
        self.latent_dim = latent_dim
        self.decoder_sizes = decoder_sizes
        self.decoder_nonlinearity = decoder_nonlinearity
        self.decoder_output_nonlinearity = decoder_output_nonlinearity
        
        # Create encoder network
        self.encoder = MLP(input_dim=self.input_dim, 
                           hidden_sizes=self.encoder_sizes, 
                           hidden_nonlinearity=self.encoder_nonlinearity, 
                           output_dim=None, 
                           output_nonlinearity=None)
        # Last layer of encoder network to output mean and log-variance for latent variable
        self.mu_head = nn.Linear(in_features=self.encoder_sizes[-1], out_features=self.latent_dim)
        self.logvar_head = nn.Linear(in_features=self.encoder_sizes[-1], out_features=self.latent_dim)
        
        # Create decoder network
        self.decoder = MLP(input_dim=self.latent_dim, 
                           hidden_sizes=self.decoder_sizes, 
                           hidden_nonlinearity=self.decoder_nonlinearity, 
                           output_dim=self.input_dim, 
                           output_nonlinearity=self.decoder_output_nonlinearity)
        
        # Initialize parameters for newly defined layers
        self._init_params()
        
    def _init_params(self):
        """
        Initialize the network parameters, weights, biases
        
        Orthogonal weight initialization and zero bias initialization
        """
        # Initialize mu_head, it does not have nonlinearity
        # Weight initialization
        nn.init.orthogonal_(self.mu_head.weight, gain=1)  # gain=1 due to identity
        # Bias initialization
        nn.init.constant_(self.mu_head.bias, 0.0)
        
        # Initialize logvar_head, it does not have nonlinearity
        # Weight initialization
        nn.init.orthogonal_(self.logvar_head.weight, gain=1)  # gain=1 due to identity
        # Bias initialization
        nn.init.constant_(self.logvar_head.bias, 0.0)
        
    def encode(self, x):
        """
        Forward pass of encoder network. 
        
        Args:
            x (Tensor): input tensor to the encoder network
            
        Returns:
            mu (Tensor): mean of the latent variable
            logvar (Tensor): log-variance of the latent variable. 
                Note that log operation allows to optimize negative values,
                though variance must be non-negative. 
        """
        x = self.encoder(x)
        # Enforce features have shape [N, D], useful for ConvVAE
        x = x.view(x.size(0), -1)
        
        # Forward pass via mu and logvar heads
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        
        return mu, logvar
    
    def decode(self, z):
        """
        Forward pass of decoder network
        
        Args:
            z (Tensor): the sampled latent variable
            
        Returns:
            x (Tensor): the reconstruction of the input
        """
        x = self.decoder(z)
        
        return x
    
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
        # Forward pass through encoder to get mu and logvar for latent variable
        mu, logvar = self.encode(x)
        # Sample latent variable by reparameterization trick
        z = self.reparameterize(mu, logvar)
        # Forward pass through decoder of sampled latent variable to reconstruct input
        reconstructed_x = self.decode(z)
        
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
                                                         reduce=False)  # all losses for [N, D]
        elif reconstruction_loss_type == 'MSE':
            reconstruction_loss = F.mse_loss(reconstructed_x, 
                                             x, 
                                             reduce=False)  # all losses for [N, D]
        # sum up loss for each data item, with shape [N]
        reconstruction_loss = reconstruction_loss.sum(1)
            
        # Calculate KL loss for each element
        # Gaussian: 0.5*sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KL_loss = -0.5*torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)
        
        # Average losses over batch
        reconstruction_loss = reconstruction_loss.mean()
        KL_loss = KL_loss.mean()
        
        # Compute total loss
        loss = reconstruction_loss + KL_loss
        
        return loss, reconstruction_loss, KL_loss