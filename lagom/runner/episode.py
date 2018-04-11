import torch

import numpy as np

from lagom.core.preprocessors import ExponentialFactorCumSum


class Episode(object):
    """
    Data for an episode, consisting of successive transitions, with additional useful information
    
    e.g. other info includes length, success and so on.
    """
    def __init__(self, gamma):
        self.gamma = gamma  # discount factor
        
        self.transitions = []
        self.info = {}
        
    def add_transition(self, transition):
        """
        Add a new transition to append in the episode
        
        Args:
            transition (Transition): given transition
        """
        self.transitions.append(transition)
        
    def add_info(self, name, value):
        """
        Add additional information for current episode
        
        Args:
            name (str): name of the information
            value (object): value of the information
        """
        self.info[name] = value
        
    @property
    def T(self):
        """
        Return the length of the episode. 
        """
        return len(self.transitions)
    
    @property
    def all_s(self):
        return [transition.s for transition in self.transitions] + [self.transitions[-1].s_next]
    
    @property
    def all_a(self):
        return [transition.a for transition in self.transitions]
    
    @property
    def all_r(self):
        return [transition.r for transition in self.transitions]
    
    @property
    def all_returns(self):
        return ExponentialFactorCumSum(self.gamma).process(self.all_r)
    
    @property
    def returns(self):
        return self.all_returns[0]
    
    @property
    def all_TD(self):
        """
        Returns all TD errors, for each time step
        """
        # Get all rewards
        all_r = np.array(self.all_r)

        # Get all state values
        all_v = self.all_info('state_value')
        # Retrive each item if dtype: torch.Tensor 
        all_v = np.array([v.item() if torch.is_tensor(v) else v for v in all_v])

        # Get all next state values, final terminal state with zero state value
        all_next_v = np.append(all_v[1:], [0])

        # Calculate TD error
        all_TD = all_r + self.gamma*all_next_v - all_v

        return all_TD.tolist()
    
    def all_info(self, name):
        """
        Return specified information for all transitions
        
        Args:
            name (str): name of the information
            
        Returns:
            list of specified information for all transitions
        """
        info = [transition.info[name] for transition in self.transitions]
        
        return info
    
    @property
    def goal_solved(self):
        """
        Return True if the final transition indicates a goal is solved. 
        """
        last_transition = self.transitions[-1]
        solved = last_transition.info.get('goal_solved', None)
        
        if solved is None:
            raise KeyError('The key of goal_solved does not exist in the final transition. ')
            
        return solved
    
    def __repr__(self):
        return str([transition.__repr__() for transition in self.transitions])