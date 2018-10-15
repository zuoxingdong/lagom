from abc import ABC
from abc import abstractmethod


class BasePlot(ABC):
    r"""Base class for plotting the experiment result. 
    
    For example, agent performance for different random runs can be plotted as a curve with uncertainty bands. 
    
    The subclass should implement at least the following:
    
    - :meth:`__call__`
    
    """
    def __init__(self):
        self.data = {}
    
    def add(self, key, value):
        r"""Add new data for plotting. 
        
        Args:
            key (str): name of the given data
            value (object): value of the given data
        """
        self.data[key] = value
        
    @abstractmethod
    def __call__(self, **kwargs):
        r"""Generate a plot. 
        
        Args:
            **kwargs: keyword aguments used to specify the plotting options. 
            
        Returns
        -------
        ax : Axes
            a matplotlib Axes representing the generated plot. 
        """
        pass
