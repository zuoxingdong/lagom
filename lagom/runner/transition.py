class Transition(object):
    """
    Data for one transition by taking an action given a state, it also contains additional useful information.
    
    i.e. one transition is of form (state, action, reward, next_state, done)
    Useful information: log-probability of action, state value etc. 
    """
    def __init__(self, s, a, r, s_next, done):
        """
        Initialize a transition
        
        Args:
            s (object): given state
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
        """
        Add additional information for current transition
        
        Args:
            name (str): name of the information
            value (object): value of the information
        """    
        self.info[name] = value
        
    @property
    def V_s(self):
        """
        Return the state value for the given state, self.s
        """
        return self.info['V_s']
    
    @property
    def V_s_next(self):
        """
        Return the state value for the next state, self.s_next. 
        If this transition leads to a terminal state, then it returns 0
        """
        # TODO: it might be memory costly to have V_s_next in each transition. 
        V_s_next = self.info['V_s_next']
        # Set to zero for terminal state
        if self.done:
            V_s_next = 0.0
        
        return V_s_next
    
    def __repr__(self):
        return f'Transition: ({self.s}, {self.a}, {self.r}, {self.s_next}, {self.done})'