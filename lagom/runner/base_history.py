class BaseHistory(object):
    r"""Base class for all history of :class:`Transition`. 
    
    It stores a list of successive transitions and a dictionary of additional useful information. 
    
    Common usecases can be :class:`Trajectory` or :class:`Segment`. 
    
    The subclass should implement at least the following:
    
    - :meth:`all_s`
    - :meth:`all_returns`
    - :meth:`all_discounted_returns`
    - :meth:`all_bootstrapped_returns`
    - :meth:`all_bootstrapped_discounted_returns`
    - :meth:`all_V`
    - :meth:`all_TD`
    - :meth:`all_GAE`
    
    """
    def __init__(self, gamma):
        r"""Initialize the history of transitions. 
        
        Args:
            gamme (float): discount factor. 
        """
        self.gamma = gamma
        
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
        
    @property
    def T(self):
        r"""Return the total number of stored transitions. """
        return len(self.transitions)
    
    @property
    def all_s(self):
        r"""Return a list of all states in the history including the terminal state. 
        
        .. note::
            
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        """
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    @property
    def all_discounted_returns(self):
        r"""Return a list of discounted returns for all time steps. 
        
        Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
        .. math::
            G_t = \sum_{i=t}^{T} \gamma^{i - t} r_i, \forall t
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
        
            It returns raw values instead of Tensor dtype, not to be used for backprop. 
            
        """
        raise NotImplementedError
        
    @property
    def all_bootstrapped_returns(self):
        r"""Return a list of accumulated returns (no discount, gamma=1.0) with bootstrapping
        for all time steps. 
        
        Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
        .. math::
            Q_t = r_t + r_{t+1} + \dots + r_T + V(s_{T+1})
        
        .. note::
        
            The state value for terminal state is set as zero !
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
        
            It returns raw values instead of Tensor dtype, not to be used for backprop. 
            
        """
        raise NotImplementedError
        
    @property
    def all_bootstrapped_discounted_returns(self):
        r"""Return a list of discounted returns with bootstrapping for all time steps. 
        
        Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
        .. math::
            Q_t = r_t + \gamma r_{t+1} + \dots + \gamma^{T - t} r_T + \gamma^{T - t + 1} V(s_{T+1})
        
        .. note::
        
            The state value for terminal state is set as zero !
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
        
            It returns raw values instead of Tensor dtype, not to be used for backprop. 
            
        """
        raise NotImplementedError
    
    @property
    def all_V(self):
        r"""Return a list of all state values in the history including the terminal states.  
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
            
            It returns Tensor dtype, used for backprop to train value function. It does not set
            zero value for terminal state !
            
        """
        raise NotImplementedError
    
    @property
    def all_TD(self):
        r"""Return a list of all TD errors in the history including the terminal states. 
        
        Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)` and all state
        values :math:`(V(s_1), \dots, V(s_T), V(s_{T+1}))`, it computes
        
        .. math::
            \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
        
        .. note::
        
            The state value for terminal state is set as zero !
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
        
            It returns raw values instead of Tensor dtype, not to be used for backprop. 
            
        """
        raise NotImplementedError
    
    def all_GAE(self, gae_lambda):
        r"""Return a list of all `generalized advantage estimates`_ (GAE) in the history including
        the terminal states.
        
        .. note::
        
            The state value for terminal state is set as zero !
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
        
            It returns raw values instead of Tensor dtype, not to be used for backprop. 
        
        .. _generalized advantage estimates:
            https://arxiv.org/abs/1506.02438
        """
        raise NotImplementedError
    
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
