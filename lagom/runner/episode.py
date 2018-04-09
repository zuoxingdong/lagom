from lagom.core.preprocessors import CalcReturn


class Episode(object):
    """
    Data for an episode, consisting of successive transitions, with additional useful information
    
    e.g. other info includes length, success and so on.
    """
    def __init__(self):
        self.transitions = []
        self.info = {}
        
    def add_transition(self, transition):
        """
        Add a new transition to append in the episode
        
        Args:
            transition (Transition): given transition
        """
        self.transitions.append(transition)
        
    def add_info(self, name, value):
        """
        Add additional information for current episode
        
        Args:
            name (str): name of the information
            value (object): value of the information
        """
        self.info[name] = value
        
    @property
    def T(self):
        """
        Return the length of the episode. 
        """
        return len(self.transitions)
    
    @property
    def all_s(self):
        return [transition.s for transition in self.transitions] + [self.transitions[-1].s_next]
    
    @property
    def all_a(self):
        return [transition.a for transition in self.transitions]
    
    @property
    def all_r(self):
        return [transition.r for transition in self.transitions]
    
    def get_returns(self, gamma):
        return CalcReturn(gamma).process(self.all_r)
    
    def all_info(self, name):
        """
        Return specified information for all transitions
        
        Args:
            name (str): name of the information
            
        Returns:
            list of specified information for all transitions
        """
        info = [transition.info.get(name, None) for transition in self.transitions]
        
        if None in info:
            raise KeyError(f'The key {name} should exist in all transition info dictionary. ')
        
        return info
    
    @property
    def goal_solved(self):
        """
        Return True if the final transition indicates a goal is solved. 
        """
        last_transition = self.transitions[-1]
        solved = last_transition.info.get('goal_solved', None)
        
        if solved is None:
            raise KeyError('The key of goal_solved does not exist in the final transition. ')
            
        return solved
    
    def __repr__(self):
        return str([transition.__repr__() for transition in self.transitions])