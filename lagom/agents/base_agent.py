class BaseAgent(object):
    r"""Base class for all agents. 
    
    The agent could select an action from a given observation and update itself by defining a certain learning
    mechanism. 
    
    Any agent should subclass this class, e.g. policy-based or value-based. 
    
    .. note::
    
        All agents should by default handle batched data e.g. batched observation returned from :class:`VecEnv`
        and batched action for each sub-environment of a :class:`VecEnv`. 
    
    The subclass should implement at least the following:
    
    - :meth:`choose_action`
    - :meth:`learn`
    - :meth:`save`
    - :meth:`load`
    
    """
    def __init__(self, config, **kwargs):
        r"""Initialize the agent. 
        
        Args:
            config (dict): a dictionary of configurations
            **kwargs: keyword aguments used to specify the agent
        """
        self.config = config
        
        # Set keyword arguments
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        
    def choose_action(self, obs):
        r"""Returns an (batched) action selected by the agent from received (batched) observation/
        
        .. note::
        
            Tensor conversion should be handled here instead of in policy or network forward pass.
        
        The output is a dictionary containing useful items, e.g. action, action_logprob, state_value
        
        Args:
            obs (object): batched observation returned from the environment. First dimension is treated
                as batch dimension. 
            
        Returns
        -------
        out : dict
            a dictionary of action selection output. It should also contain all useful information
            to be stored during interaction with :class:`BaseRunner`. This allows a generic API of
            the runner classes for all kinds of agents. Note that everything should be batched even
            if for scalar loss, i.e. ``scalar_loss -> [scalar_loss]``
        """
        raise NotImplementedError
        
    def learn(self, D):
        r"""Defines learning mechanism to update the agent from a batched data. 
        
        Args:
            D (list): a list of batched data to train the agent e.g. in policy gradient, this can be 
                a list of :class:`Trajectory` or :class:`Segment`
            
        Returns
        -------
        out : dict
            a dictionary of learning output. This could contain the loss. 
        """
        raise NotImplementedError
        
    def save(self, f):
        r"""Save the current parameters of the agent. 
        
        If the agent uses a :class:`BaseNetwork`, it is recommended to call its internal
        ``save`` function to serialize its parameters. 
        
        Args:
            f (str): name of the file
        """
        raise NotImplementedError
        
    def load(self, f):
        r"""Load the parameters of the agent from a file.
        
        If the agent uses a :class:`BaseNetwork`, it is recommended to call its internal
        ``load`` function to load network parameters. 
        
        Args:
            f (str): name of the file
        """
        raise NotImplementedError
