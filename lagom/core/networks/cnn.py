import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class CNN(nn.Module):
    """
    Convolutional neural networks. 
    """
    def __init__(self, 
                 input_channel, 
                 input_shape,
                 conv_kernels,
                 conv_kernel_sizes,
                 conv_strides, 
                 conv_pads, 
                 conv_nonlinearity, 
                 hidden_sizes=None, 
                 hidden_nonlinearity=None, 
                 output_dim=None, 
                 output_nonlinearity=None):
        """
        Set up CNN with configurations. 
        
        Args:
            input_channel (int): the number of channels of the input, e.g. color channel
            input_shape (list): [Height, Width] of the input, to calculate the 
            conv_kernels (list): a list of number of kernels (filters or feature maps), for each convolutional layer. 
            conv_kernel_sizes (list): a list of kernel sizes, [int or tuple], for each convolutional layer. 
            conv_strides (list): a list of strides, for each convolutional layer. 
            conv_pads (list): a list of paddings, for each convolutional layer. 
            conv_nonlinearity (nn.functional): nonlinearity for convolutional layers
            hidden_sizes (list): a list of sizes for hidden layers
            hidden_nonlinearity (nn.functional): nonlinearity for hidden layers
            output_dim (int): output dimension
                                If None, then no output layer to be generated (useful if output has different heads)
            output_nonlinearity (nn.functional): nonlinearity for output layer
                                If None, then no output nonlinearity
        """
        super().__init__()
        
        self.input_channel = input_channel
        self.input_shape = input_shape
        self.conv_kernels = conv_kernels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.conv_pads = conv_pads 
        self.conv_nonlinearity = conv_nonlinearity
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_dim = output_dim
        self.output_nonlinearity = output_nonlinearity
        
        # Iteratively build convolutional layers
        # Should use nn.Sequential, otherwise cannot be recognized
        self.conv_layers = nn.Sequential()
        # Augment the input channel to the list of conv_kernels
        kernels = [self.input_channel] + self.conv_kernels
        for i in range(len(self.conv_kernels)):
            in_channels = kernels[i]
            out_channels = kernels[i+1]
            conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=self.conv_kernel_sizes[i], 
                             stride=self.conv_strides[i], 
                             padding=self.conv_pads[i])
            self.conv_layers.add_module(f'Conv {i+1}', conv)

        # MLP
        if self.hidden_sizes is not None:
            # Temporary forward pass to determine MLP input_dim
            input_dim = self._conv_out_dim()
            # Create MLP
            self.MLP = MLP(input_dim=input_dim,
                           hidden_sizes=self.hidden_sizes, 
                           hidden_nonlinearity=self.hidden_nonlinearity,
                           output_dim=self.output_dim, 
                           output_nonlinearity=self.output_nonlinearity)
        
        # Initialize parameters
        self._init_params()
        
    def forward(self, x):
        # Forward pass through convolutional layers
        x = self._conv_forward(x)
        
        if self.hidden_sizes is not None:
            # Flatten the tensors for MLP
            x = x.view(x.size(0), -1)
            
            # Forward pass through MLP
            x = self.MLP(x)
            
        return x
        
    def _conv_forward(self, x):
        for conv in self.conv_layers:
            x = self.conv_nonlinearity(conv(x))
            
        return x
    
    def _conv_out_dim(self):
        x = torch.zeros(1, self.input_channel, *self.input_shape)
        x = self._conv_forward(x)
        out_dim = x.view(1, -1).size(1)
        
        return out_dim
        
    def _init_params(self):
        """
        Initialize the parameters for convolutional layers, filter weights and biases
        
        Orthogonal weight initialization and zero bias initialization
        """
        # TODO: more flexible initialization API
        # Iterate over all convolutional layers
        for conv in self.conv_layers:
            # Calculate gain for the nonlinearity
            gain = nn.init.calculate_gain(self.conv_nonlinearity.__name__)
            # Weight initialization
            nn.init.orthogonal_(conv.weight, gain=gain)
            # Bias initialization
            nn.init.constant_(conv.bias, 0.0)