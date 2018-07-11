import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNetwork


class BaseVAE(BaseNetwork):
    """
    Base class for variational autoencoders (VAE), works for both MLP and CNN versions.
    
    All inherited subclass should implement the following functions
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
    def __init__(self, config=None):
        # Override this constructor
        super(BaseNetwork, self).__init__()  # call nn.Module.__init__()
        
        self.config = config
        
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


'''
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
        
        # Compute total loss
        losses = reconstruction_loss + KL_loss
        
        # Average losses over batch
        loss = losses.mean()
        
        return loss, reconstruction_loss.mean(), KL_loss.mean()
        
        
        
        
        
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import CNN
from .transposed_cnn import TransposedCNN
from .vae import VAE


class ConvVAE(VAE):
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
                for final decoder transposed convolutional layer. e.g. sigmoid for BCE loss
        """
        super(VAE, self).__init__()  # call __init__ in nn.Module
        
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
        
        # Create latent variable
        # Last layer of encoder network to output mean and log-variance for latent variable
        if self.encoder_hidden_sizes is None:  # Encoder directly map Conv features to latent variable
            in_features = self.encoder._calculate_conv_flatten_dim()
        else:  # Encoder has fully connected layers with Conv features before mapping to latent variable
            in_features = self.encoder_hidden_sizes[-1]
        self.mu_head = nn.Linear(in_features=in_features, 
                                 out_features=self.latent_dim)
        self.logvar_head = nn.Linear(in_features=in_features, 
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
        super()._init_params()
'''