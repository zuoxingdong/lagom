import torch

import numpy as np

from lagom.core.transform import ExpFactorCumSum

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
        [1, 2, 3, 4]
        
        >>> trajectory.all_r
        [0.5, 0.5, 1.0]
        
        >>> trajectory.all_done
        [False, False, True]
        
        >>> trajectory.all_V
        [10.0, 20.0, 30.0, 40.0]
        
        >>> trajectory.all_bootstrapped_returns
        [2.0, 1.5, 1.0]

        >>> trajectory.all_discounted_returns
        [0.56, 0.6, 1.0]

        >>> trajectory.all_TD
        [-7.5, -16.5, -29.0]
    
    """
    def add_transition(self, transition):
        # Sanity check for trajectory
        # Not allowed to add more transition if it already contains done=True
        if len(self.transitions) > 0:  # non-empty
            assert not self.transitions[-1].done, 'not allowed to add transition, because already contains done=True'
        super().add_transition(transition)
    
    @property
    def all_s(self):
        r"""Return a list of all states in the trajectory, from first state to the last state (i.e. ``.s_next`` in 
        last transition). 
        """
        return [transition.s for transition in self.transitions] + [self.transitions[-1].s_next]
    
    @property
    def all_returns(self):
        return ExpFactorCumSum(1.0)(self.all_r)
    
    @property
    def all_discounted_returns(self):
        return ExpFactorCumSum(self.gamma)(self.all_r)
    
    def _rewards_with_bootstrapping(self):
        # Get last state value and last done
        last_V = self.transitions[-1].V_s_next
        last_done = self.transitions[-1].done
        # Get raw value if Tensor dtype
        if torch.is_tensor(last_V):
            last_V = last_V.item()
        assert isinstance(last_V, float), f'expected float dtype, got {type(last_V)}'
        
        # Set zero value if terminal state
        if last_done:
            last_V = 0.0
            
        return self.all_r + [last_V]
    
    @property
    def all_bootstrapped_returns(self):
        bootstrapped_rewards = self._rewards_with_bootstrapping()
        
        out = ExpFactorCumSum(1.0)(bootstrapped_rewards)
        # Take out last one, because it is just last state value itself
        out = out[:-1]
        
        return out
        
    @property
    def all_bootstrapped_discounted_returns(self):
        bootstrapped_rewards = self._rewards_with_bootstrapping()
        
        out = ExpFactorCumSum(self.gamma)(bootstrapped_rewards)
        # Take out last one, because it is just last state value itself
        out = out[:-1]
        
        return out
    
    @property
    def all_V(self):
        return [transition.V_s for transition in self.transitions] + [self.transitions[-1].V_s_next]
    
    @property
    def all_TD(self):
        # Get all rewards
        all_r = np.array(self.all_r)
        
        # Get all state values with raw values if Tensor dtype
        all_V = np.array([v.item() if torch.is_tensor(v) else v for v in self.all_V])
        # Set last state value as zero if terminal state
        if self.all_done[-1]:
            all_V[-1] = 0.0
        
        # Unpack state values into current and next time step
        all_V_s = all_V[:-1]
        all_V_s_next = all_V[1:]
        
        # Calculate TD error
        all_TD = all_r + self.gamma*all_V_s_next - all_V_s
        
        return all_TD.astype(np.float32).tolist()
    
    def all_GAE(self, gae_lambda):
        # TODO: implement it + add to test_runner
        raise NotImplementedError
