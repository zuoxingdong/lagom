import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters
from torch.nn.utils import parameters_to_vector


class Module(nn.Module):
    r"""Wrap PyTorch nn.module to provide more helper functions. """
    def __init__(self, **kwargs):
        super().__init__()
        
        for key, val in kwargs.items():
            self.__setattr__(key, val)
        
    @property
    def num_params(self):
        r"""Returns the total number of parameters in the neural network. """
        return sum(param.numel() for param in self.parameters())
        
    @property
    def num_trainable_params(self):
        r"""Returns the total number of trainable parameters in the neural network."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
    
    @property
    def num_untrainable_params(self):
        r"""Returns the total number of untrainable parameters in the neural network. """
        return sum(param.numel() for param in self.parameters() if not param.requires_grad)
    
    def to_vec(self):
        r"""Returns the network parameters as a single flattened vector. """
        return parameters_to_vector(parameters=self.parameters())
    
    def from_vec(self, x):
        r"""Set the network parameters from a single flattened vector.
        
        Args:
            x (Tensor): A single flattened vector of the network parameters with consistent size.
        """
        vector_to_parameters(vec=x, parameters=self.parameters())
    
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
        torch.save(obj=self.state_dict(), f=f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        
    def load(self, f):
        r"""Load the network parameters from a file. 
        
        It complies with the `recommended approach for saving a model in PyTorch documentation`_. 
        
        Args:
            f (str): file path. 
            
        .. _recommended approach for saving a model in PyTorch documentation:
            https://pytorch.org/docs/master/notes/serialization.html#best-practices
        """
        self.load_state_dict(torch.load(f))
