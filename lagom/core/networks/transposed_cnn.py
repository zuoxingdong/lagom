import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class TransposedCNN(nn.Module):
    """
    Transposed convolutional neural networks. 
    """
    def __init__(self, 
                 input_dim,
                 hidden_sizes,
                 hidden_nonlinearity, 
                 conv_transposed_input_channel, 
                 conv_transposed_input_shape,
                 conv_transposed_kernels,
                 conv_transposed_kernel_sizes,
                 conv_transposed_strides, 
                 conv_transposed_pads, 
                 conv_transposed_nonlinearity, 
                 output_nonlinearity=None):
        """
        Set up tranposed CNN with configurations. 
        
        Args:
            input_dim (int): input dimension for the first fully connected layer
            hidden_sizes (list): a list of sizes for hidden layers
            hidden_nonlinearity (nn.functional): nonlinearity for hidden layers
            conv_transposed_input_channel (int): the number of channels to feed into transposed convolutional layers
            conv_transposed_input_shape (list): [Height, Width] of the input tensor to feed into transposed convolutional layers
            conv_transposed_kernels (list): a list of number of kernels (filters or feature maps), 
                for each transposed convolutional layer. 
            conv_transposed_kernel_sizes (list): a list of kernel sizes, [int or tuple], 
                for each transposed convolutional layer. 
            conv_transposed_strides (list): a list of strides, for each transposed convolutional layer. 
            conv_transposed_pads (list): a list of paddings, for each transposed convolutional layer. 
            conv_transposed_nonlinearity (nn.functional): nonlinearity for transposed convolutional layers
            output_nonlinearity (nn.functional): nonlinearity for final transposed convolutional layer
        """
        super().__init__()
        
        # Enforce the consistency of final hidden size and conv_transposed input channel and input shape
        assert hidden_sizes[-1] == \
            conv_transposed_input_channel*conv_transposed_input_shape[0]*conv_transposed_input_shape[1]
        
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        
        self.conv_transposed_input_channel = conv_transposed_input_channel
        self.conv_transposed_input_shape = conv_transposed_input_shape
        
        self.conv_transposed_kernels = conv_transposed_kernels
        self.conv_transposed_kernel_sizes = conv_transposed_kernel_sizes
        self.conv_transposed_strides = conv_transposed_strides
        self.conv_transposed_pads = conv_transposed_pads
        self.conv_transposed_nonlinearity = conv_transposed_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        
        # MLP
        self.MLP = MLP(input_dim=self.input_dim, 
                       hidden_sizes=self.hidden_sizes, 
                       hidden_nonlinearity=self.hidden_nonlinearity, 
                       output_dim=None, 
                       output_nonlinearity=None)
        
        # Iteratively build transposed convolutional layers
        # Should use nn.Sequential, otherwise cannot be recognized
        self.conv_transposed_layers = nn.Sequential()
        # Augment the input channel to the list of conv_transposed_kernels
        kernels = [self.conv_transposed_input_channel] + self.conv_transposed_kernels
        for i in range(len(self.conv_transposed_kernels)):
            in_channels = kernels[i]
            out_channels = kernels[i+1]
            conv_transposed = nn.ConvTranspose2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=self.conv_transposed_kernel_sizes[i], 
                                                 stride=self.conv_transposed_strides[i], 
                                                 padding=self.conv_transposed_pads[i])
            self.conv_transposed_layers.add_module(f'Transposed Conv {i+1}', conv_transposed)

        # Initialize parameters
        self._init_params()
        
    def forward(self, x):
        # Forward pass through MLP
        x = self.MLP(x)
        
        # Reshape the tensor consistent with transposed convolutional layers
        x = x.view(-1, self.conv_transposed_input_channel, *self.conv_transposed_input_shape)
        
        # Forward pass through all transposed convolutional layers
        for i, conv_transposed in enumerate(self.conv_transposed_layers):
            if i < len(self.conv_transposed_layers) - 1:  # before final layer
                x = self.conv_transposed_nonlinearity(conv_transposed(x))
            else:  # final layer
                x = conv_transposed(x)
                if self.output_nonlinearity is not None:
                    x = self.output_nonlinearity(x)
        
        return x
        
    def _init_params(self):
        """
        Initialize the parameters for transposed convolutional layers, filter weights and biases
        
        Orthogonal weight initialization and zero bias initialization
        """
        # TODO: more flexible initialization API
        # Iterate over all transposed convolutional layers
        for conv_transposed in self.conv_transposed_layers:
            # Calculate gain for the nonlinearity
            gain = nn.init.calculate_gain(self.conv_transposed_nonlinearity.__name__)
            # Weight initialization
            nn.init.orthogonal_(conv_transposed.weight, gain=gain)
            # Bias initialization
            nn.init.constant_(conv_transposed.bias, 0.0)