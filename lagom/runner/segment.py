import torch

import numpy as np

from lagom.core.transform import ExpFactorCumSum


class Segment(object):
    """
    Segment of interactions between agent and environment. 
    It consists of a sequence of transitions.
    
    Note that it can contain sequence of transitions from multiple episodes successively in general. 
    
    A special case can be a trajectory such that the sequence of transitions are coming from 
    part of a single episode. 
    
    The number of episodes' data contained can be clarifed by looking at list of done in all transitions:
    We split with following cases when value of done, e.g. for a segment with length 6
    1. Part of single episode: [False, False, False, False, False]
    2. Part of single episode with final terminal transition: [False, False, False, False, False, True]
    3. Parts of two episodes (first episode terminates): [False, False, True, False, False]
    4. Parts of two episodes (both episodes terminate): [False, False, True, False, True]
    
    Some common properties for such transitions are provided as class method. 
    i.e. all states, all actions, all state values, all TD errors etc. 
    """
    def __init__(self, gamma):
        """
        Args:
            gamma (float): discounted factor
        """
        self.gamma = gamma
        
        self.transitions = []
        self.info = {}  # some information about the segment
        
    def add_transition(self, transition):
        """
        Add a new transition to append in the segment
        
        Args:
            transition (Transition): given transition
        """
        self.transitions.append(transition)
        
    def add_info(self, name, value):
        """
        Add additional information for the segment
        
        Args:
            name (str): name of the information
            value (object): value of the information
        """
        self.info[name] = value
        
    @property
    def split_transitions(self):
        """
        Return a list of splitted transitions for different episodes contained in the segment. 
        If there are only single episodic transitions, the length of the list is one. 
        """
        split_transitions = []
        
        # Get all indices where done=True
        done_idx = np.where(self.all_done)[0]
        
        # start index for slicing
        start_idx = 0
        # Iterate over all done index to slice, note it is ok with empty done_idx
        for idx in done_idx:
            # Record the sliced transitions, note python slicing can deal with final index
            split_transitions.append(self.transitions[start_idx : idx+1])
            # Update the start index
            start_idx = idx + 1
        # Record the last part if still remaining
        if start_idx < self.T:
            split_transitions.append(self.transitions[start_idx:])
        
        return split_transitions

    @property
    def T(self):
        """
        Return the length of the segment (number of transitions). 
        """
        return len(self.transitions)
    
    @property
    def T_split(self):
        """
        Return a list of length of transitions in splitted segment. 
        """
        T_split = [len(transitions) for transitions in self.split_transitions]
        
        assert sum(T_split) == self.T
        
        return T_split
    
    @property
    def all_s(self):
        """
        Return a list of all states in the segment. 
        """
        # done counter for sanity check
        done_counter = 0
        
        all_s = []
        # Iterate over all transitions until the last second
        for transition in self.transitions[:-1]:
            all_s.append(transition.s)
            # Record s_next for each done=True, i.e. episodic terminal state
            if transition.done:
                all_s.append(transition.s_next)
                done_counter += 1
        # Record the final transition for both s and s_next
        all_s.append(self.transitions[-1].s)
        all_s.append(self.transitions[-1].s_next)
        
        # sanity check
        assert len(all_s) == len(self.transitions) + done_counter + 1
        
        return all_s
    
    @property
    def all_s_split(self):
        """
        Return a list of splitted episodic states. Each contain successive states from each episode. 
        """
        all_s_split = []
        
        # Get all_s
        all_s = self.all_s
        
        start_idx = 0
        # Iterate over splitted length of episodic transitions
        for T in self.T_split:
            end_idx = start_idx + T + 1  # plus one: inclusive slicing
            # Record sliced states
            all_s_split.append(all_s[start_idx:end_idx])
            # Update start index
            start_idx += T + 1
            
        return all_s_split

    @property
    def all_a(self):
        """
        Return a list of all actions in the segment. 
        """
        return [transition.a for transition in self.transitions]
    
    @property
    def all_r(self):
        """
        Return a list of all rewards in the segment. 
        """
        return [transition.r for transition in self.transitions]
    
    @property
    def all_done(self):
        """
        Return a list of all dones in the segment. 
        """
        return [transition.done for transition in self.transitions]
    
    @property
    def all_returns(self):
        r"""
        Return a list of returns (no discount, gamma=1.0) for all time steps. 
        
        Suppose we have all rewards [r_1, ..., r_T], it computes
        G_t = \sum_{i=t}^{T} r_i
        
        TODO: change this description
        Note that it is okay when the history is a segment with transitions from several
        episodes, in other words, some done is True for intermediate time steps and
        transitions for new episode follows afterwards. 
        
        This can be generally computed by using a binary mask. Set one for `done=False`
        and set zero for `done=True`. Thus each time reaching a zero value, the returns
        will be recalculated from that time step on. 
        """
        return self._compute_all_returns(use_discount=False)
    
    @property
    def all_discounted_returns(self):
        """
        Return a list of discounted returns for all time steps. 
        
        Suppose we have all rewards [r_1, ..., r_T], it computes
        G_t = \sum_{i=t}^{T} \gamma^{i - t} r_i
        
        Note that it is okay when the history is a segment with transitions from several
        episodes, in other words, some done is True for intermediate time steps and
        transitions for new episode follows afterwards. 
        
        This can be generally computed by using a binary mask. Set one for `done=False`
        and set zero for `done=True`. Thus each time reaching a zero value, the returns
        will be recalculated from that time step on. 
        """
        return self._compute_all_returns(use_discount=True)
    
    def _compute_all_returns(self, use_discount):
        """
        Helper function for shared computation between `all_returns` and `all_discounted_returns`
        
        Args:
            use_discount (bool): Whether to use discount factor
        """
        mask = np.logical_not(self.all_done).astype(int).tolist()
        
        if use_discount:
            gamma = self.gamma
        else:
            gamma = 1.0
            
        return ExpFactorCumSum(gamma)(self.all_r, mask=mask)
    
    @property
    def all_bootstrapped_returns(self):
        r"""
        Return a list of boostrapped returns (no discount, gamma=1.0) for all time steps. 
        
        Suppose we have all rewards [r_1, ..., r_T] and final state value V(s_T), it computes
        G_t = \sum_{i=t}^{T} r_i + V(s_T)
        
        TODO: change this description
        Note that it is okay when the history is a segment with transitions from several
        episodes, in other words, some done is True for intermediate time steps and
        transitions for new episode follows afterwards. 
        
        This can be generally computed by using a binary mask. Set one for `done=False`
        and set zero for `done=True`. Thus each time reaching a zero value, the returns
        will be recalculated from that time step on. 
        """
        return self._compute_bootstrapped_returns(use_discount=False)
    
    @property
    def all_bootstrapped_discounted_returns(self):
        r"""
        Return a list of boostrapped discounted returns for all time steps. 
        
        Suppose we have all rewards [r_1, ..., r_T] and final state value V(s_T), it computes
        G_t = \sum_{i=t}^{T} \gamma^{i - t} r_i + \gamma^{T - t + 1} V(s_T)
        
        TODO: change this description
        Note that it is okay when the history is a segment with transitions from several
        episodes, in other words, some done is True for intermediate time steps and
        transitions for new episode follows afterwards. 
        
        This can be generally computed by using a binary mask. Set one for `done=False`
        and set zero for `done=True`. Thus each time reaching a zero value, the returns
        will be recalculated from that time step on. 
        """
        return self._compute_bootstrapped_returns(use_discount=True)
    
    def _compute_bootstrapped_returns(self, use_discount):
        """
        Helper function for shared computation between `all_bootstrapped_returns` 
        and `all_bootstrapped_discounted_returns`
        
        Args:
            use_discount (bool): Whether to use discount factor
        """
        # Get all rewards
        all_r = np.array(self.all_r)
        
        # Get all done
        all_done = np.array(self.all_done)
        # Make binary mask: set 0 for done=True and set 1 for done=False
        mask = np.logical_not(all_done).astype(int)
        
        # Compute all indicies with done=True except for last transition
        dones_idx = np.where(all_done[:-1])[0]
        # Augment all_r and mask with 0.0 followed by all dones_idx
        # We do this because we want fast vectorized computation
        # It will be reverted (remove additional position) after further computation
        all_r = np.insert(all_r, dones_idx+1, 0.0)  # r_terminal = 0.0
        mask = np.insert(mask, dones_idx+1, 0.0)  # mask_terminal = 0.0
        
        # Augment all_r and mask for last transition
        V_final = self.transitions[-1].V_s_next
        if torch.is_tensor(V_final):
            V_final = V_final.item()
        # Add final state value for bootstrapping
        all_r = np.append(all_r, V_final)
        # Add final mask, 0.0
        mask = np.append(mask, 0.0)
        
        # Compute bootstrapped returns
        if use_discount:
            gamma = self.gamma
        else:
            gamma = 1.0
        G = ExpFactorCumSum(gamma)(all_r, mask=mask.tolist())
        
        # Remove the augmented values
        delete_idx = dones_idx + np.arange(1, len(dones_idx)+1, dtype=np.float32)
        G = np.delete(G, delete_idx)
        # Remove the final one
        G = np.delete(G, -1)
        
        return G.astype(np.float32).tolist()
    
    @property
    def all_V(self):
        """
        Return a list of all state values, from first to last state in an episode. 
        
        It takes information with the key 'V_s' for all transitions
        and augment it with 'V_s_next' of the last transition. 
        
        Note that we would like to keep Tensor dtype, used for backprop.
        
        Note that it is okay when the history is a segment with transitions from several
        episodes, in other words, some done is True for intermediate time steps and
        transitions for new episode follows afterwards. 
        
        This can be generally computed by using a binary mask. Set one for `done=False`
        and set zero for `done=True`. Thus each time reaching a zero value, the returns
        will be recalculated from that time step on. 
        """
        # done counter for sanity check
        done_counter = 0
        
        all_V = []
        
        # Iterate over all transitions until the last second
        for transition in self.transitions[:-1]:  # except for last one
            # Record V_s
            all_V.append(transition.V_s)
            
            # Record V_s_next for each done=True, i.e. episodic terminal state, for backprop of zero target
            if transition.done:
                all_V.append(transition.V_s_next)
                done_counter += 1
        # Record both V_s and V_s_next for final transition
        all_V.append(self.transitions[-1].V_s)
        all_V.append(self.transitions[-1].V_s_next)
        
        # sanity check
        assert len(all_V) == len(self.transitions) + done_counter + 1

        return all_V
    
    @property
    def all_TD(self):
        r"""
        Return a list of TD errors for all time steps. 
        
        It requires that each transition has the information with key 'V_s' and
        last transition in an episode with both 'V_s' and 'V_s_next'.
        
        If done=True, then V_s_next should be zero as terminal state value. 
        
        TD error is calculated as following:
        \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
        
        Note that we would like to use raw float dtype, rather than Tensor. 
        Because we often do not backprop via TD error. 
        
        Note that it is okay when the history is a segment with transitions from several
        episodes, in other words, some done is True for intermediate time steps and
        transitions for new episode follows afterwards. 
        
        This can be generally computed by using a binary mask. Set one for `done=False`
        and set zero for `done=True`. Thus each time reaching a zero value, the returns
        will be recalculated from that time step on. 
        """
        # Get all rewards
        all_r = np.array(self.all_r)
        
        # Get all state values
        # Retrieve raw value if dtype is Tensor, because we often do not backprop TD error
        all_V = np.array([v.item() if torch.is_tensor(v) else v for v in self.all_V])
        
        # Get all done
        all_done = np.array(self.all_done)
        # Make binary mask: set 0 for done=True and set 1 for done=False
        mask = np.logical_not(all_done).astype(int)
        
        # Compute all indicies with done=True except for last transition
        # Because no matter the final done is True or False, it will be handled by mask automatically
        # And we always have state value V_s_next for final transition regardless of done
        dones_idx = np.where(all_done[:-1])[0]
        # Augment all_r and mask with 0.0 followed by all dones_idx
        # We do this because we want fast vectorized computation of TD
        # the additional computation will be deleted in final step
        all_r = np.insert(all_r, dones_idx+1, 0.0)  # r_terminal = 0.0
        mask = np.insert(mask, dones_idx+1, 0.0)  # mask_terminal = 0.0
        
        # Unpack the state values as current and next time step by shifting one position to the right
        all_V_s = all_V[:-1]
        all_V_s_next = all_V[1:]
        
        # Calculate TD error for each time step
        # Note that we use binary mask to calculate next state value, to deal with terminal state (zero)
        all_TD = all_r + self.gamma*all_V_s_next*mask - all_V_s
        
        # Remove the augmented computation for intermediate done=True (except for final transition)
        # Note dones_idx can be empty
        delete_idx = dones_idx + np.arange(1, len(dones_idx)+1, dtype=np.float32)
        all_TD = np.delete(all_TD, delete_idx)
        
        return all_TD.astype(np.float32).tolist()

    def all_gae(self, gae_lambda):
        """
        Return a list of GAE. 
        https://arxiv.org/abs/1506.02438
        
        Args:
            gae_lambda (float): GAE lambda, in range [0, 1]
        
        TODO: remaining work. 
        """
        raise NotImplementedError
        
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
