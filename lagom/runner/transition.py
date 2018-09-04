class Transition(object):
    r"""Define one transition from current state to next state by taking an action. 
    
    Formally, it stores the transition tuple :math:`(s, a, r, s_{\text{next}}, \text{done})`. 
    
    It also stores additional useful information, e.g. log-probability of action, state value etc. 
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
              or :class:`Segment`
        
        Args:
            name (str): name of the information
            value (object): value of the information
        """    
        self.info[name] = value
        
    @property
    def V_s(self):
        r"""Return the state value for the given state, self.s"""
        return self.info['V_s']
    
    @property
    def V_s_next(self):
        """
        Return the state value for the next state, self.s_next. 
        
        Note that we do not set zero value for terminal state, because this function
        often returns the Tensor dtype, used for backprop. 
        
        Please be careful when using returned values in later phase !
        """
        # TODO: it might be memory costly to have V_s_next in each transition. 
        return self.info['V_s_next']
    
    def __repr__(self):
        return f'Transition: ({self.s}, {self.a}, {self.r}, {self.s_next}, {self.done})'
