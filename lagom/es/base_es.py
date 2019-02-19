from abc import ABC
from abc import abstractmethod


class BaseES(ABC):
    r"""Base class for all evolution strategies. 
    
    .. note::
    
        The optimization is treated as minimization. e.g. maximize rewards is equivalent to minimize negative rewards.
    
    """
    @abstractmethod
    def ask(self):
        r"""Sample a set of new candidate solutions. 
        
        Returns
        -------
        solutions : list
            sampled candidate solutions
        """
        pass
        
    @abstractmethod
    def tell(self, solutions, function_values):
        r"""Update the parameters of the population for a new generation based on the values of the objective
        function evaluated for sampled solutions. 
        
        Args:
            solutions (list/ndarray): candidate solutions returned from :meth:`ask`
            function_values (list): a list of objective function values evaluated for the sampled solutions.
        """
        pass
        
    @property
    @abstractmethod
    def result(self):
        r"""Return a dictionary of all necessary results for the optimization. 
        
        Returns
        -------
        results : dict
            a dictionary of results e.g. ['best_param', 'best_f_val', 'hist_best_param', 'stds']
        """
        pass
