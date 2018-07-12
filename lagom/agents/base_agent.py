class BaseAgent(object):
    """
    Base class of the agent for action selection and learning rule. 
    
    Depending on the type of agent (e.g. policy-based or value-based), it is recommended
    to override the constructor __init__() to provide essential items for the agent. 
    
    All inherited subclasses should implement the following functions:
    1. choose_action(self, obs)
    2. learn(self, x)
    3. save(self, filename)
    4. load(self, filename)
    
    Note that if subclass overrides __init__, remember to provide
    keywords aguments, i.e. **kwargs passing to super().__init__. 
    """
    def __init__(self, config, **kwargs):
        """
        Args:
            config (dict): the configurations
            **kwargs: TODO for documentation
        """
        self.config = config
        
        # Set all keyword arguments
        for key, val in kwargs.items():
            self.__setattr__(key, val)
        
    def choose_action(self, obs):
        """
        The agent selects an action based on given observation. 
        The output is a dictionary containing useful items, e.g. action, action_logprob, state_value
        
        Args:
            obs (object): agent's observation. Note that this is raw observation returned from 
                environment. Tensor conversion should be handled here. 
            
        Returns:
            output (dict): a dictionary of action selection output. 
                Possible keys: ['action', 'action_logprob', 'state_value', 'Q_value']
        """
        raise NotImplementedError
        
    def learn(self, x):
        """
        Learning rule about how agent updates itself given data.
        The output is a dictionary containing useful items, i.e. loss, batched_policy_loss
        
        Args:
            x (object): input data to train the agent. 
                e.g. In policy gradient, this can be a list of episodes
            
        Returns:
            output (dict): a dictionary of learning output. 
                Possible keys: ['loss', 'batch_policy_loss']
        """
        raise NotImplementedError
        
    def save(self, filename):
        """
        Save the current parameters of the agent. 
        
        If the agent uses a BaseNetwork object, it is recommended to call
        BaseNetwork internal save/load function for network parameters in PyTorch. 
        
        Args:
            filename (str): name of the file
        """
        raise NotImplementedError
        
    def load(self, filename):
        """
        Load the parameters of the agent from a file
        
        If the agent uses a BaseNetwork object, it is recommended to call
        BaseNetwork internal save/load function for network parameters in PyTorch. 
        
        Args:
            filename (str): name of the file
        
        """
        raise NotImplementedError