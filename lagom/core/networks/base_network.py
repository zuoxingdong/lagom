import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector


class BaseNetwork(nn.Module):
    r"""Base class for all neural networks. 
    
    Any neural network should subclass this class. 
    
    The subclass should implement at least the following:
    
    - :meth:`make_params`
    - :meth:`init_params`
    - :meth:`forward`
    
    Example::
    
        import torch.nn as nn
        import torch.nn.functional as F
        from lagom.core.networks import BaseNetwork


        class Network(BaseNetwork):
            def make_params(self, config):
                self.fc1 = nn.Linear(3, 2)
                self.fc2 = nn.Linear(2, 1)

            def init_params(self, config):
                gain = nn.init.calculate_gain('relu')

                nn.init.orthogonal_(self.fc1.weight, gain=gain)
                nn.init.constant_(self.fc1.bias, 0.0)

                nn.init.orthogonal_(self.fc2.weight, gain=gain)
                nn.init.constant_(self.fc2.bias, 0.0)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)

                return x

    """
    def __init__(self, config=None, **kwargs):
        r"""Initialize the neural network.
        
        Args:
            config (dict): A dictionary of configurations. 
            **kwargs: keyword arguments to specify the network. 
        """
        super().__init__()
        
        self.config = config
        
        # Set all keyword arguments
        for key, val in kwargs.items():
            self.__setattr__(key, val)
    
        # Create all trainable parameters/layers
        self.make_params(self.config)
        
        # Initialize all created parameters/layers
        self.init_params(self.config)
        
    def make_params(self, config):
        r"""Create all trainable parameters/layers for the neural network according to 
        a given configuration. 
        
        .. note::
        
            All created layers must be assigned as a class attributes to be automatically
            tracked. e.g. ``self.fc = nn.Linear(3, 2)``. 
        
        Args:
            config (dict): a dictionary of configurations. 
        """
        raise NotImplementedError
        
    def init_params(self, config):
        r"""Initialize all created parameters in :meth:`make_params` according to a 
        given configuration. 
        
        Args:
            config (dict): a dictionary of configurations. 
        """
        raise NotImplementedError
        
    @property
    def num_params(self):
        r"""Returns the total number of trainable parameters in the neural network."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
    
    def save(self, f):
        r"""Save the network parameters to a file. 
        
        It complies with the `recommended approach for saving a model in PyTorch documentation`_. 
        
        .. note::
            It uses the highest pickle protocol to serialize the network parameters. 
        
        Args:
            f (str): file path. 
            
        .. _recommended approach for saving a model in PyTorch documentation:
            https://pytorch.org/docs/master/notes/serialization.html#best-practices
        """
        import pickle
        torch.save(self.state_dict(), f, pickle.HIGHEST_PROTOCOL)
        
    def load(self, f):
        r"""Load the network parameters from a file. 
        
        It complies with the `recommended approach for saving a model in PyTorch documentation`_. 
        
        Args:
            f (str): file path. 
            
        .. _recommended approach for saving a model in PyTorch documentation:
            https://pytorch.org/docs/master/notes/serialization.html#best-practices
        """
        self.load_state_dict(torch.load(f))
        
    def to_vec(self):
        r"""Returns the network parameters as a single flattened vector. """
        return parameters_to_vector(parameters=self.parameters())
    
    def from_vec(self, x):
        r"""Set the network parameters from a single flattened vector.
        
        Args:
            x (Tensor): A single flattened vector of the network parameters with consistent size.
        """
        vector_to_parameters(vec=x, parameters=self.parameters())
