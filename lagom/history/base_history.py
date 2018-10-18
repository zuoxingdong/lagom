from abc import ABC
from abc import abstractmethod


class BaseHistory(ABC):
    r"""Base class for all history of :class:`Transition`. 
    
    It stores a list of successive transitions and a dictionary of additional useful information. 
    
    Common usecases can be :class:`Trajectory` or :class:`Segment`. 
    
    The subclass should implement at least the following:
    
    - :meth:`all_s`
    - :meth:`all_returns`
    - :meth:`all_discounted_returns`
    
    """
    def __init__(self):
        self.transitions = []
        self.info = {}
        
    def add_transition(self, transition):
        r"""Append a new transition. 
        
        Args:
            transition (Transition): a transition. 
        """
        self.transitions.append(transition)
        
    def add_info(self, name, value):
        r"""Add additional information. 
        
        Args:
            name (str): name of the information
            value (object): value of the information
        """
        self.info[name] = value
        
    def get_info(self, name):
        r"""Returns the information given the name. 
        
        Args:
            name (str): name of the information
        """
        return self.info[name]
        
    @property
    def T(self):
        r"""Return the total number of stored transitions. """
        return len(self.transitions)
    
    @property
    @abstractmethod
    def all_s(self):
        r"""Return a list of all states in the history and the final state separately. 
        
        .. note::
            
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        """
        pass
    
    @property
    def all_a(self):
        r"""Return a list of all actions in the history."""
        return [transition.a for transition in self.transitions]
    
    @property
    def all_r(self):
        r"""Return a list of all rewards in the history."""
        return [transition.r for transition in self.transitions]
    
    @property
    def all_done(self):
        r"""Return a list of all dones in the history."""
        return [transition.done for transition in self.transitions]
    
    @property
    @abstractmethod
    def all_returns(self):
        r"""Return a list of accumulated returns (no discount, gamma=1.0) for all time steps. 
        
        Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
        .. math::
            G_t = \sum_{i=t}^{T} r_i, \forall t
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
        
            It returns raw values instead of Tensor dtype, not to be used for backprop. 
            
        """
        pass
    
    @abstractmethod
    def all_discounted_returns(self, gamma):
        r"""Return a list of discounted returns for all time steps. 
        
        Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
        .. math::
            G_t = \sum_{i=t}^{T} \gamma^{i - t} r_i, \forall t
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
        
            It returns raw values instead of Tensor dtype, not to be used for backprop. 
            
        """
        pass
        
    def all_info(self, name):
        r"""Return the specified information from all transitions.
        
        Args:
            name (str): name of the information
            
        Returns
        -------
        list
            a list of the specified information from all transitions
        """
        info = [transition.info[name] for transition in self.transitions]
        
        return info
    
    def __repr__(self):
        string = f'{self.__class__.__name__}: \n'
        for transition in self.transitions:
            string += '\t' + transition.__repr__() + '\n'
        return string
