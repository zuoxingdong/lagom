from abc import ABC
from abc import abstractmethod

from lagom.nn import Module


class BaseAgent(Module, ABC):
    r"""Base class for all agents. 
    
    The agent could select an action from some data (e.g. observation) and update itself by
    defining a certain learning mechanism. 
    
    Any agent should subclass this class, e.g. policy-based or value-based. 
    
    Args:
        config (Config): a dictionary of configurations
        env (Env): environment object. 
        **kwargs: keyword aguments used to specify the agent
    
    """
    def __init__(self, config, env, **kwargs):
        super(Module, self).__init__(**kwargs)
        
        self.config = config
        self.env = env
        
        self.info = {}
        self.is_recurrent = None
        
    @abstractmethod
    def choose_action(self, x, **kwargs):
        r"""Returns the selected action given the data.
        
        .. note::
        
            It's recommended to handle all dtype/device conversions between CPU/GPU or Tensor/Numpy here.
        
        The output is a dictionary containing useful items, 
        
        Args:
            obs (object): batched observation returned from the environment. First dimension is treated
                as batch dimension. 
            **kwargs: keyword arguments to specify action selection.
            
        Returns:
            dict: a dictionary of action selection output. It contains all useful information (e.g. action, 
                action_logprob, state_value). This allows the API to be generic and compatible with
                different kinds of runner and agents. 
        """
        pass
        
    @abstractmethod
    def learn(self, D, **kwargs):
        r"""Defines learning mechanism to update the agent from a batched data. 
        
        Args:
            D (list): a list of batched data to train the agent e.g. in policy gradient, this can be 
                a list of :class:`Trajectory`.
            **kwargs: keyword arguments to specify learning mechanism
            
        Returns:
            dict: a dictionary of learning output. This could contain the loss and other useful metrics. 
        """
        pass


class RandomAgent(BaseAgent):
    r"""A random agent samples action uniformly from action space. """    
    def choose_action(self, x, **kwargs):
        action = self.env.action_space.sample()
        out = {'raw_action': action}
        return out  

    def learn(self, D, **kwargs):
        pass
