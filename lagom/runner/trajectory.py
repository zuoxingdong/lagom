import torch

import numpy as np

from lagom.core.transform import ExpFactorCumSum

from .base_history import BaseHistory


class Trajectory(BaseHistory):
    r"""Define a trajectory, consisting of successive transitions. 
    
    .. note::
    
        It is not necessarily a complete episode (final state is terminal state). However, all transitions
        must come from a single episode. For the history containing transitions from multiple episodes 
        (i.e. ``done=True`` in the middle), it is recommended to use :class:`Segment` instead. 
    
    Example::
    
    """
    def add_transition(self, transition):
        # Sanity check for trajectory
        # Not allowed to add more transition if it already contains done=True
        if len(self.transitions) > 0:  # non-empty
            assert self.transitions[-1].done == False, 'not allowed to add transition, because already contains done=True'
        super().add_transition(transition)
    
    @property
    def all_s(self):
        r"""Return a list of all states in the trajectory, from first state to the last state (i.e. ``s_next`` in 
        last transition). 
        """
        return [transition.s for transition in self.transitions] + [self.transitions[-1].s_next]
    
    @property
    def all_returns(self):
        return ExpFactorCumSum(1.0)(self.all_r)
    
    @property
    def all_discounted_returns(self):
        return ExpFactorCumSum(self.gamma)(self.all_r)
    
    @property
    def all_V(self):
        return [transition.V_s for transition in self.transitions] + [self.transitions[-1].V_s_next]
    
    @property
    def all_TD(self):
        r"""
        Return a list of TD errors for all time steps. 
        
        It requires that each transition has the information with key 'V_s' and
        last transition with both 'V_s' and 'V_s_next'.
        
        If last transition with done=True, then V_s_next should be zero as terminal state value. 
        
        TD error is calculated as following:
        \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
        
        Note that we would like to use raw float dtype, rather than Tensor. 
        Because we often do not backprop via TD error. 
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
        
        return all_TD.astype(np.float32).tolist()
    
    @property
    def all_GAE(self, gae_lambda):
        raise NotImplementedError
