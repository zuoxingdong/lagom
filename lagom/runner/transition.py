class Transition(object):
    r"""Define one transition from current state to next state by taking an action. 
    
    Formally, it stores the transition tuple :math:`(s, a, r, s_{\text{next}}, \text{done})`. 
    
    It also stores additional useful information, e.g. log-probability of action, state value etc. 
    
    Example::
    
        >>> transition = Transition(s=0.2, a=1.3, r=-1.0, s_next=0.8, done=True)
        >>> transition
        Transition: (s=0.2, a=1.3, r=-1.0, s_next=0.8, done=True)
        
    """
    def __init__(self, s, a, r, s_next, done):
        r"""Initialize the transition
        
        Args:
            s (object): current state
            a (object): action
            r (float): reward
            s_next (object): next state
            done (bool): indicate if an episode ends. 
        """
        self.s = s
        self.a = a
        self.r = float(r)  # ensure it is float for PyTorch compatibility. Often r from gym environment as scalar value. 
        self.s_next = s_next
        self.done = done
        
        # Additional information
        self.info = {}
        
    def add_info(self, name, value):
        r"""Add additional information for the transition. 
        
        .. note::
        
            For certain information, the specific key is required. It shows as following
            
            * ``'V_s'``: state value for current state (i.e. :attr:`s`)
            
            * ``'V_s_next'``: state value for next state (i.e. :attr:`s_next`). Note that
              it should only be added for the last transition in either :class:`Trajectory`
              or :class:`Segment` to save memory. 
        
        Args:
            name (str): name of the information
            value (object): value of the information
        """    
        self.info[name] = value
        
    @property
    def V_s(self):
        r"""Return the state value for the current state, i.e. :attr:`s`"""
        return self.info['V_s']
    
    @property
    def V_s_next(self):
        r"""Return the state value for the next state, i.e. :attr:`s_next` 
        
        .. note::
        
            Often it returns as Tensor dtype, it can be useful for backprop to train
            value function. However, be cautious of handling the raw value e.g. calculate
            bootstrapped returns, then zero value should be replaced when the next state 
            is terminal state. 
        """
        return self.info['V_s_next']
    
    def __repr__(self):
        return f'Transition: (s={self.s}, a={self.a}, r={self.r}, s_next={self.s_next}, done={self.done})'
