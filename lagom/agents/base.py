class BaseAgent(object):
    """
    Base class for the agent for action selection and learning rule. 
    """
    def __init__(self, config):
        """
        Args:
            config (dict): all configurations
        """
        self.config = config
        
    def choose_action(self, obs):
        """
        The agent selects an action based on given observation. 
        The output is a dictionary containing useful items, e.g. action, logprob_action, state_value
        
        Args:
            obs (object): agent's observation
            
        Returns:
            output (dict): a dictionary with possible keys 
                            e.g. ['action', 'logprob_action', 'state_value']
        """
        raise NotImplementedError
        
    def learn(self, batch):
        """
        Learning rule about how agent updates itself given a batch of data.
        The output is a dictionary containing useful items, i.e. total_loss, batched_policy_loss
        
        Args:
            batch (list of Episode): a list of episodes to train the agent
        Returns:
            output (dict): a dictionary with possible keys
                            e.g. [total_loss, batched_policy_loss]
        """
        raise NotImplementedError
        
    def save(self, filename):
        """
        Save the current parameters of the agent. 
        
        Args:
            filename (str): name of the file
        """
        raise NotImplementedError
        
    def load(self, filename):
        """
        Load the parameters of the agent from a file
        
        Args:
            filename (str): name of the file
        
        """
        raise NotImplementedError