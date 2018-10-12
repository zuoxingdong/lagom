from abc import ABC
from abc import abstractmethod


class BaseAlgorithm(ABC):
    r"""Base class for all algorithms.
    
    Any algorithm should subclass this class. 
    
    The subclass should implement at least the following:
    
    - :meth:`__call__`
    
    """ 
    @abstractmethod
    def __call__(self, config, seed, device):
        r"""Run the algorithm with a configuration, a random seed and a PyTorch device.
        
        Args:
            config (dict): a dictionary of configuration items
            seed (int): a random seed to run the algorithm
            device (torch.device): a PyTorch device. 
            
        Returns
        -------
        out : object
            output of the algorithm execution. If no need to return anything, then an ``None`` should be returned. 
        """
        pass
