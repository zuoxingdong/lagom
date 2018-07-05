from .base_network import BaseNetwork


class BaseCNN(BaseNetwork):
    """
    Base class for convolutional neural networks. 
    
    All inherited subclass should implement the following functions
    1. make_params(self, config)
    2. init_params(self, config)
    3. forward(self, x)
    
    Examples:
    
    class CNN(BaseCNN):
        def make_params(self, config):
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3)
            self.fc1 = nn.Linear(in_features=4608, out_features=1)

        def init_params(self, config):
            gain = nn.init.calculate_gain('relu')

            nn.init.orthogonal_(self.conv1.weight, gain=gain)
            nn.init.constant_(self.conv1.bias, 0.0)

            nn.init.orthogonal_(self.conv2.weight, gain=gain)
            nn.init.constant_(self.conv2.bias, 0.0)

            nn.init.orthogonal_(self.fc1.weight, gain=gain)
            nn.init.constant_(self.fc1.bias, 0.0)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.shape[0], -1)
            print(x.shape)
            x = self.fc1(x)

            return x
    """
    def __init__(self, config=None):
        super().__init__(config)

'''
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
            input_shape (list): [Height, Width] of the input
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
            input_dim = self._calculate_conv_flatten_dim()
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
        """
        Forward pass of all convolutional layers
        
        Args:
            x (Tensor): input tensor
        
        Returns:
            output Tensor of final convolutional layer (after nonlinearity)
        """
        for conv in self.conv_layers:
            x = self.conv_nonlinearity(conv(x))
            
        return x
    
    # TODO: use with torch.no_grad to save memory
    def _calculate_conv_flatten_dim(self):
        """
        Calculate the flattened dimension of the output of last convolutional layer via a temporary forward pass
        The flattend dimension is used to define the first fully connected layer
        
        Returns:
            dim (int): the flattened dimension of the output of last convolutional layer
        """
        # Create a temporary input tensor, i.e. zero tensor
        x = torch.zeros(1, self.input_channel, *self.input_shape)
        # Forward pass via all convolutional layers
        x = self._conv_forward(x)
        # Calculate the flattened dimension
        dim = x.view(1, -1).size(1)
        
        return dim
        
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


'''