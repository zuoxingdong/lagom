from .base_network import BaseNetwork


class BaseMLP(BaseNetwork):
    """
    Base class for fully connected neural network (or Multi-Layer Perceptron)
    
    All inherited subclass should implement the following functions
    1. make_params(self, config)
    2. init_params(self, config)
    3. forward(self, x)
    
    Examples: 
    
    class MLP(BaseMLP):
        def make_params(self, config):
            self.fc1 = nn.Linear(in_features=5, out_features=32)
            self.fc2 = nn.Linear(in_features=32, out_features=10)

        def init_params(self, config):
            gain = nn.init.calculate_gain(nonlinearity='relu')

            nn.init.orthogonal_(self.fc1.weight, gain=gain)
            nn.init.constant_(self.fc1.bias, 0.0)

            nn.init.orthogonal_(self.fc2.weight, gain=gain)
            nn.init.constant_(self.fc2.bias, 0.0)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

            return x
    """
    def __init__(self, config=None):
        super().__init__(config)



'''
class MLP(nn.Module):
    """
    Fully connected neural network (or Multi-Layer Perceptron)
    """
    def __init__(self, 
                 input_dim,
                 hidden_sizes, 
                 hidden_nonlinearity, 
                 output_dim=None, 
                 output_nonlinearity=None):
        """
        Set up MLP with configurations
        
        Args:
            input_dim (int): input dimension
            hidden_sizes (list): a list of sizes for hidden layers
            hidden_nonlinearity (nn.functional): nonlinearity for hidden layers
            output_dim (int): output dimension
                                If None, then no output layer to be generated (useful if output has different heads)
            output_nonlinearity (nn.functional): nonlinearity for output layer
                                If None, then no output nonlinearity
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_dim = output_dim
        self.output_nonlinearity = output_nonlinearity
        
        # Iteratively build network based on the sizes of hidden layers
        # Should use nn.Sequential, otherwise cannot be recognized
        self.hidden_layers = nn.Sequential()
        # Augment the input dimension to the list of hidden sizes
        sizes = [self.input_dim] + self.hidden_sizes
        for i in range(len(self.hidden_sizes)):
            in_features = sizes[i]
            out_features = sizes[i+1]
            self.hidden_layers.add_module(f'FC {i+1}', nn.Linear(in_features, out_features))
        # Output layer
        if self.output_dim is not None:
            self.output_layer = nn.Linear(self.hidden_sizes[-1], self.output_dim)
            
        # Initialize parameters
        self._init_params()
        
    def forward(self, x):
        # Enforce the shape of x to be consistent with first layer
        x = x.view(-1, self.input_dim)
        
        # Forward pass till last hidden layer
        for layer in self.hidden_layers:
            x = self.hidden_nonlinearity(layer(x))
        
        # Output layer
        if self.output_dim is not None:
            x = self.output_layer(x)
            if self.output_nonlinearity is not None:
                x = self.output_nonlinearity(x)
                
        return x
    
    def _init_params(self):
        """
        Initialize the network parameters, weights, biases
        
        Orthogonal weight initialization and zero bias initialization
        """
        # Initialization for hidden layers
        for layer in self.hidden_layers:
            # Calculate gain for the nonlinearity
            gain = nn.init.calculate_gain(self.hidden_nonlinearity.__name__)
            # Weight initialization
            nn.init.orthogonal_(layer.weight, gain=gain)
            # Bias initialization
            nn.init.constant_(layer.bias, 0.0)
            
        # Initialization for output layer
        if self.output_dim is not None:  # existence of output layer
            if self.output_nonlinearity is not None:  # existence of output nonlinearity
                # Calculate gain for the nonlinearity
                gain = nn.init.calculate_gain(self.output_nonlinearity.__name__)
                # Weight initialization
                nn.init.orthogonal_(self.output_layer.weight, gain=gain)
            else:  # identity, no nonlinearity
                # Weight initialization
                nn.init.orthogonal_(self.output_layer.weight, gain=1)  # gain=1 due to identity
            
            # Bias initialization
            nn.init.constant_(self.output_layer.bias, 0.0)
'''