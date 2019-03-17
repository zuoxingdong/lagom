from abc import ABC
from abc import abstractmethod

from lagom.networks import Module
from lagom.envs import VecEnv


class BaseAgent(Module, ABC):
    r"""Base class for all agents. 
    
    The agent could select an action from a given observation and update itself by defining a certain learning
    mechanism. 
    
    Any agent should subclass this class, e.g. policy-based or value-based. 
    
    .. note::
    
        All agents should by default handle batched data e.g. batched observation returned from :class:`VecEnv`
        and batched action for each sub-environment of a :class:`VecEnv`. 
    
    Args:
        config (dict): a dictionary of configurations
        env (VecEnv): environment object. 
        device (Device): a PyTorch device
        **kwargs: keyword aguments used to specify the agent
    
    """
    def __init__(self, config, env, device, **kwargs):
        super(Module, self).__init__(**kwargs)
        
        self.config = config
        self.env = env
        self.device = device
        
        self.info = {}
        self.is_recurrent = None
        
    @abstractmethod
    def choose_action(self, obs, **kwargs):
        r"""Returns an (batched) action selected by the agent from received (batched) observation/
        
        .. note::
        
            Tensor conversion should be handled here instead of in policy or network forward pass.
        
        The output is a dictionary containing useful items, e.g. action, action_logprob, state_value
        
        Args:
            obs (object): batched observation returned from the environment. First dimension is treated
                as batch dimension. 
            **kwargs: keyword arguments to specify action selection.
            
        Returns
        -------
        out : dict
            a dictionary of action selection output. It should also contain all useful information
            to be stored during interaction with :class:`BaseRunner`. This allows a generic API of
            the runner classes for all kinds of agents. Note that everything should be batched even
            if for scalar loss, i.e. ``scalar_loss -> [scalar_loss]``
        """
        pass
        
    @abstractmethod
    def learn(self, D, **kwargs):
        r"""Defines learning mechanism to update the agent from a batched data. 
        
        Args:
            D (list): a list of batched data to train the agent e.g. in policy gradient, this can be 
                a list of :class:`Trajectory` or :class:`Segment`
            **kwargs: keyword arguments to specify learning mechanism
            
        Returns
        -------
        out : dict
            a dictionary of learning output. This could contain the loss. 
        """
        pass


class RandomAgent(BaseAgent):
    r"""A random agent samples action uniformly from action space. """    
    def choose_action(self, obs, **kwargs):
        if isinstance(self.env, VecEnv):
            action = [self.env.action_space.sample() for _ in range(len(self.env))]
        else:
            action = self.env.action_space.sample()
        out = {'raw_action': action}
        return out  

    def learn(self, D, **kwargs):
        pass
