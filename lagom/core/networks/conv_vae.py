import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import CNN
from .transposed_cnn import TransposedCNN


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoders (Conv-VAE)
    """
    def __init__(self, 
                 input_channel,
                 input_shape,
                 encoder_conv_kernels,
                 encoder_conv_kernel_sizes,
                 encoder_conv_strides, 
                 encoder_conv_pads, 
                 encoder_conv_nonlinearity, 
                 encoder_hidden_sizes, 
                 encoder_hidden_nonlinearity, 
                 latent_dim, 
                 decoder_hidden_sizes,
                 decoder_hidden_nonlinearity, 
                 decoder_conv_trans_input_channel,
                 decoder_conv_trans_input_shape,
                 decoder_conv_trans_kernels,
                 decoder_conv_trans_kernel_sizes,
                 decoder_conv_trans_strides, 
                 decoder_conv_trans_pads, 
                 decoder_conv_trans_nonlinearity, 
                 decoder_output_nonlinearity=None):
        """
        Set up Convolutional VAE with configurations. 
        
        Args:
            input_channel (int): the number of channels of the input, e.g. color channel
            input_shape (list): [Height, Width] of the input
            encoder_conv_kernels (list): a list of number of kernels (filters or feature maps), 
                for each encoder convolutional layer. 
            encoder_conv_kernel_sizes (list): a list of kernel sizes, [int or tuple], 
                for each encoder convolutional layer. 
            encoder_conv_strides (list): a list of strides, for each encoder convolutional layer. 
            encoder_conv_pads (list): a list of paddings, for each encoder convolutional layer. 
            encoder_conv_nonlinearity (nn.functional): nonlinearity for encoder convolutional layers. 
            encoder_hidden_sizes (list): a list of sizes for encoder hidden layers
            encoder_hidden_nonlinearity (nn.functional): nonlinearity for encoder hidden layers
            latent_dim (int): latent dimension
            decoder_hidden_sizes (list): a list of sizes for decoder hidden layers
            decoder_hidden_nonlinearity (nn.functional): nonlinearity for decoder hidden layers
            decoder_conv_trans_input_channel (int): the number of channels to feed into 
                transposed convolutional layers
            decoder_conv_trans_input_shape (list): [Height, Width] of the input tensor to feed into 
                transposed convolutional layers
            decoder_conv_trans_kernels (list): a list of number of kernels (filters or feature maps), 
                for each decoder transposed convolutional layer. 
            decoder_conv_trans_kernel_sizes (list): a list of kernel sizes, [int or tuple], 
                for each decoder transposed convolutional layer. 
            decoder_conv_trans_strides (list): a list of strides, 
                for each decoder transposed convolutional layer. 
            decoder_conv_trans_pads (list): a list of paddings, 
                for each decoder transposed convolutional layer. 
            decoder_conv_trans_nonlinearity  (nn.functional): nonlinearity 
                for each decoder transposed convolutional layer. 
            decoder_output_nonlinearity (nn.functional): nonlinearity 
                for final decoder transposed convolutional layer. 
        """
        super().__init__()
        
        self.input_channel = input_channel
        self.input_shape = input_shape
        self.encoder_conv_kernels = encoder_conv_kernels
        self.encoder_conv_kernel_sizes = encoder_conv_kernel_sizes
        self.encoder_conv_strides = encoder_conv_strides
        self.encoder_conv_pads = encoder_conv_pads
        self.encoder_conv_nonlinearity = encoder_conv_nonlinearity
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.encoder_hidden_nonlinearity = encoder_hidden_nonlinearity
        self.latent_dim = latent_dim
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.decoder_hidden_nonlinearity = decoder_hidden_nonlinearity
        self.decoder_conv_trans_input_channel = decoder_conv_trans_input_channel
        self.decoder_conv_trans_input_shape = decoder_conv_trans_input_shape
        self.decoder_conv_trans_kernels = decoder_conv_trans_kernels
        self.decoder_conv_trans_kernel_sizes = decoder_conv_trans_kernel_sizes
        self.decoder_conv_trans_strides = decoder_conv_trans_strides
        self.decoder_conv_trans_pads = decoder_conv_trans_pads
        self.decoder_conv_trans_nonlinearity = decoder_conv_trans_nonlinearity
        self.decoder_output_nonlinearity = decoder_output_nonlinearity
        
        # Create encoder network
        self.encoder = CNN(input_channel=self.input_channel, 
                           input_shape=self.input_shape,
                           conv_kernels=self.encoder_conv_kernels,
                           conv_kernel_sizes=self.encoder_conv_kernel_sizes,
                           conv_strides=self.encoder_conv_strides, 
                           conv_pads=self.encoder_conv_pads, 
                           conv_nonlinearity=self.encoder_conv_nonlinearity, 
                           hidden_sizes=self.encoder_hidden_sizes, 
                           hidden_nonlinearity=self.encoder_hidden_nonlinearity, 
                           output_dim=None, 
                           output_nonlinearity=None)
        # Last layer of encoder network to output mean and log-variance for latent variable
        self.mu_head = nn.Linear(in_features=self.encoder_hidden_sizes[-1], 
                                 out_features=self.latent_dim)
        self.logvar_head = nn.Linear(in_features=self.encoder_hidden_sizes[-1], 
                                     out_features=self.latent_dim)
        
        # Create decoder network
        self.decoder = TransposedCNN(input_dim=self.latent_dim,
                                     hidden_sizes=self.decoder_hidden_sizes,
                                     hidden_nonlinearity=self.decoder_hidden_nonlinearity, 
                                     conv_transposed_input_channel=self.decoder_conv_trans_input_channel, 
                                     conv_transposed_input_shape=self.decoder_conv_trans_input_shape,
                                     conv_transposed_kernels=self.decoder_conv_trans_kernels,
                                     conv_transposed_kernel_sizes=self.decoder_conv_trans_kernel_sizes,
                                     conv_transposed_strides=self.decoder_conv_trans_strides, 
                                     conv_transposed_pads=self.decoder_conv_trans_pads, 
                                     conv_transposed_nonlinearity=self.decoder_conv_trans_nonlinearity, 
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
        # Use sigmoid to constraint all values in (0, 1)
        x = F.sigmoid(x)
        
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
    
    def calculate_loss(self, reconstructed_x, x, mu, logvar):
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
        
        Returns:
            loss (Tensor): VAE loss
        """
        # Enforce the shape of x is the same as reconstructed x
        x = x.view_as(reconstructed_x)
        
        # Calculate reconstruction loss
        reconstruction_loss = F.binary_cross_entropy(reconstructed_x, 
                                                     x, 
                                                     size_average=False)  # summed up losses
        # Calculate KL loss
        # Gaussian: 0.5*sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KL_loss = -0.5*torch.sum(1 + logvar - mu**2 - logvar.exp())
        
        loss = reconstruction_loss + KL_loss
        
        return loss