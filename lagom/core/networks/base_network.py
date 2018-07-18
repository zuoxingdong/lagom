import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector


class BaseNetwork(nn.Module):
    """
    Base class for neural networks. 
    
    Depending on the type of neural networks (e.g. policy network, Q-network), it is recommended
    to override the constructor __init__ to provide essential items for the neural network. 
    
    Note that if subclass overrides __init__, remember to provide
    keywords aguments, i.e. **kwargs passing to super().__init__. 
    
    All inherited subclasses should at least implement the following functions:
    1. make_params(self, config)
    2. init_params(self, config)
    """
    def __init__(self, config=None, **kwargs):
        super().__init__()
        
        self.config = config
        
        # Set all keyword arguments
        for key, val in kwargs.items():
            self.__setattr__(key, val)
    
        # User-defined function to create all trainable parameters (layers)
        self.make_params(self.config)
        
        # User-defined function to initialize all created parameters
        self.init_params(self.config)
        
    def make_params(self, config):
        """
        User-defined function to create all trainable parameters (layers)
        
        Args:
            config (Config): configurations
            
        Examples:
            Refer to each inherited subclass with individual documentation. 
        """
        raise NotImplementedError
        
    def init_params(self, config):
        """
        User-defined function to initialize all created parameters
        
        Args:
            config (Config): configurations
        """
        raise NotImplementedError
        
    @property
    def num_params(self):
        """
        Returns the number of trainable parameters. 
        """
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
    
    def save(self, f):
        """
        Save the model parameters. It is saved by using recommended way from PyTorch documentation. 
        https://pytorch.org/docs/master/notes/serialization.html#best-practices
        
        Args:
            f (str): saving path
        """
        torch.save(self.state_dict(), f)
        
    def load(self, f):
        """
        Load the model parameters. It is loaded by using recommended way from PyTorch documentation. 
        https://pytorch.org/docs/master/notes/serialization.html#best-practices
        
        Args:
            f (str): loading path
        """
        self.load_state_dict(torch.load(f))
        
    def to_vec(self):
        """
        Flatten the network parameters into a single big vector. 
        """
        return parameters_to_vector(parameters=self.parameters())
    
    def from_vec(self, x):
        """
        Unflatten the given vector as the network parameters. 
        
        Args:
            x (Tensor): flattened single vector with size consistent of the number of network paramters. 
        """
        vector_to_parameters(vec=x, parameters=self.parameters())
