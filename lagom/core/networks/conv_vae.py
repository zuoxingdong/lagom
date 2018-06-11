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