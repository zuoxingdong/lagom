from abc import ABC
from abc import abstractmethod

from .module import Module


class BaseNetwork(Module, ABC):
    r"""Base class for all neural networks. 
    
    Any neural network should subclass this class. 
    
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
    def __init__(self, config=None, device=None, **kwargs):
        r"""Initialize the neural network.
        
        Args:
            config (dict): A dictionary of configurations. 
            device (device): a PyTorch device for this network. 
            **kwargs: keyword arguments to specify the network. 
        """
        super().__init__(**kwargs)
        
        self.config = config
        self.device = device
        
        self.make_params(self.config)
        
        self.init_params(self.config)
        
        self.reset(self.config)
        
        self.to(self.device)
        
    @abstractmethod
    def make_params(self, config):
        r"""Create all trainable parameters/layers for the neural network according to 
        a given configuration. 
        
        .. note::
        
            All created layers must be assigned as a class attributes to be automatically
            tracked. e.g. ``self.fc = nn.Linear(3, 2)``. 
        
        Args:
            config (dict): a dictionary of configurations. 
        """
        pass
        
    @abstractmethod
    def init_params(self, config):
        r"""Initialize all created parameters in :meth:`make_params` according to a 
        given configuration. 
        
        Args:
            config (dict): a dictionary of configurations. 
        """
        pass
    
    
    @abstractmethod
    def reset(self, config, **kwargs):
        r"""Reset the network.
        
        For example, this can be used for resetting the hidden state for recurrent neural networks. 
        
        Args:
            config (dict): a dictionary of configurations. 
            **kwargs: keyword arguments to specify reset function. 
        """
        pass
