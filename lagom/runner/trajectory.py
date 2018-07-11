import torch

import numpy as np

from lagom.core.transform import ExpFactorCumSum


class Trajectory(object):
    """
    Data for a trajectory, consisting of successive transitions, with additional useful information
    e.g. other info includes length, success and so on.
    
    Note that it is not necessarily an episode (with terminal state). It can be a segment of temporal
    transitions.
    """
    def __init__(self, gamma):
        self.gamma = gamma  # discount factor
        
        self.transitions = []
        self.info = {}
        
    def add_transition(self, transition):
        """
        Add a new transition to append in the trajectory
        
        Args:
            transition (Transition): given transition
        """
        self.transitions.append(transition)
        
    def add_info(self, name, value):
        """
        Add additional information for current trajectory
        
        Args:
            name (str): name of the information
            value (object): value of the information
        """
        self.info[name] = value
        
    @property
    def T(self):
        """
        Return the current length of the trajectory (number of transitions). 
        """
        return len(self.transitions)
    
    @property
    def all_s(self):
        """
        Return a list of all states in the trajectory from initial state to last state. 
        """
        return [transition.s for transition in self.transitions] + [self.transitions[-1].s_next]
    
    @property
    def all_a(self):
        """
        Return a list of all actions in the trajectory. 
        """
        return [transition.a for transition in self.transitions]
    
    @property
    def all_r(self):
        """
        Return a list of all rewards in the trajectory. 
        """
        return [transition.r for transition in self.transitions]
    
    @property
    def all_done(self):
        """
        Return a list of all dones in the trajectory. 
        
        Note that the done for initial state is not included. 
        """
        return [transition.done for transition in self.transitions]
    
    @property
    def all_returns(self):
        r"""
        Return a list of returns (no discount, gamma=1.0) for all time steps. 
        
        Suppose we have all rewards [r_1, ..., r_T], it computes
        G_t = \sum_{i=t}^{T} r_i
        """
        return ExpFactorCumSum(1.0)(self.all_r)
    
    @property
    def all_discounted_returns(self):
        """
        Return a list of discounted returns for all time steps. 
        
        Suppose we have all rewards [r_1, ..., r_T], it computes
        G_t = \sum_{i=t}^{T} \gamma^{i - t} r_i
        """
        return ExpFactorCumSum(self.gamma)(self.all_r)
    
    @property
    def all_V(self):
        """
        Return a list of all state values, from first to last state. 
        
        It takes information with the key 'V_s' for all transitions
        and augment it with 'V_s_next' of the last transition. 
        """
        return [transition.V_s for transition in self.transitions] + [self.transitions[-1].V_s_next]
    
    @property
    def all_TD(self):
        r"""
        Returns a list of TD errors for all time steps. 
        
        It requires that each transition has the information with key 'V_s' and
        last transition with both 'V_s' and 'V_s_next'.
        
        If last transition with done=True, then V_s_next should be zero as terminal state value. 
        
        TD error is calculated as following:
        \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
        """
        # Get all rewards
        all_r = np.array(self.all_r)
        
        # Get all state values
        # Retrieve raw value if dtype is Tensor
        all_V = np.array([v.item() if torch.is_tensor(v) else v for v in self.all_V])
        if self.all_done[-1]:  # value of terminal state is zero
            assert all_V[-1] == 0.0
        
        # Unpack state values into current and next time step
        all_V_s = all_V[:-1]
        all_V_s_next = all_V[1:]
        
        # Calculate TD error
        all_TD = all_r + self.gamma*all_V_s_next - all_V_s
        
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
    
    def __repr__(self):
        return str([transition.__repr__() for transition in self.transitions])