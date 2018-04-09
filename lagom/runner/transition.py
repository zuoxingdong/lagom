class Transition(object):
    """
    Data for one transition by taking an action given a state, it also contains additional useful information.
    
    i.e. one transition is of form (state, action, reward, next_state)
    Useful information: log-probability of action, state value prediction, done and so on. 
    """
    def __init__(self, s, a, r, s_next):
        """
        Initialize a transition
        
        Args:
            s (object): given state
            a (object): action
            r (float): reward
            s_next (object): next state
        """
        self.s = s
        self.a = a
        self.r = r
        self.s_next = s_next
        
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
    
    def __repr__(self):
        return f'Transition: ({self.s}, {self.a}, {self.r}, {self.s_next})'