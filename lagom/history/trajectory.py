import torch

import numpy as np

from lagom.transform import ExpFactorCumSum

from .base_history import BaseHistory


class Trajectory(BaseHistory):
    r"""Define a trajectory of successive transitions from a single episode. 
    
    .. note::
    
        It is not necessarily a complete episode (final state is terminal state). However, all transitions
        must come from a single episode. For the history containing transitions from multiple episodes 
        (i.e. ``done=True`` in the middle), it is recommended to use :class:`Segment` instead. 
    
    Example::
    
        >>> from lagom.runner import Transition
        >>> transition1 = Transition(s=1, a=0.1, r=0.5, s_next=2, done=False)
        >>> transition1.add_info(name='V_s', value=10.0)

        >>> transition2 = Transition(s=2, a=0.2, r=0.5, s_next=3, done=False)
        >>> transition2.add_info(name='V_s', value=20.0)

        >>> transition3 = Transition(s=3, a=0.3, r=1.0, s_next=4, done=True)
        >>> transition3.add_info(name='V_s', value=30.0)
        >>> transition3.add_info(name='V_s_next', value=40.0)

        >>> trajectory = Trajectory(gamma=0.1)
        >>> trajectory.add_transition(transition1)
        >>> trajectory.add_transition(transition2)
        >>> trajectory.add_transition(transition3)
        
        >>> trajectory
        Trajectory: 
            Transition: (s=1, a=0.1, r=0.5, s_next=2, done=False)
            Transition: (s=2, a=0.2, r=0.5, s_next=3, done=False)
            Transition: (s=3, a=0.3, r=1.0, s_next=4, done=True)
        
        >>> trajectory.all_s
        ([1, 2, 3], 4)
        
        >>> trajectory.all_r
        [0.5, 0.5, 1.0]
        
        >>> trajectory.all_done
        [False, False, True]
        
        >>> trajectory.all_V
        ([10.0, 20.0, 30.0], [40.0, True])
        
        >>> trajectory.all_bootstrapped_returns
        [2.0, 1.5, 1.0]

        >>> trajectory.all_discounted_returns
        [0.56, 0.6, 1.0]

        >>> trajectory.all_TD
        [-7.5, -16.5, -29.0]
    
    """
    def add_transition(self, transition):
        assert not self.complete, 'not allowed to add transition, because already contains done=True'
        super().add_transition(transition)
    
    @property
    def all_s(self):
        return [transition.s for transition in self.transitions], self.transitions[-1].s_next
    
    @property
    def all_returns(self):
        return ExpFactorCumSum(1.0)(self.all_r)
    
    def all_discounted_returns(self, gamma):
        return ExpFactorCumSum(gamma)(self.all_r)
    
    @property
    def complete(self):
        if len(self.transitions) == 0:
            return False
        else:
            return self.transitions[-1].done
