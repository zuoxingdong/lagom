from abc import ABC
from abc import abstractmethod

from lagom.networks.module import Module


class BaseAgent(Module, ABC):
    r"""Base class for all agents. 
    
    The agent could select an action from a given observation and update itself by defining a certain learning
    mechanism. 
    
    Any agent should subclass this class, e.g. policy-based or value-based. 
    
    .. note::
    
        All agents should by default handle batched data e.g. batched observation returned from :class:`VecEnv`
        and batched action for each sub-environment of a :class:`VecEnv`. 
    
    """
    def __init__(self, config, env_spec, device, **kwargs):
        r"""Initialize the agent. 
        
        Args:
            config (dict): a dictionary of configurations
            env_spec (EnvSpec): environment specification. 
            device (Device): a PyTorch device
            **kwargs: keyword aguments used to specify the agent
        """
        super(Module, self).__init__(**kwargs)
        
        self.config = config
        self._env_spec = env_spec
        self.device = device
        
        self.info = {}
        
        self.make_modules(self.config)
        
        self.prepare(self.config)
        
    def add_info(self, name, value):
        r"""Add internal extra information for the agent. 
        
        Args:
            name (str): name of the information
            value (object): value of the information
        """
        self.info[name] = value
        
    def get_info(self, name):
        r"""Returns the information given the name. 
        
        Args:
            name (str): name of the information
        """
        return self.info[name]
        
    @abstractmethod
    def make_modules(self, config):
        r"""Create all modules for the agent. 
        
        For example, this allows to easily create either policy network or Q-network here.
        
        .. note::
        
            All created networks must be assigned as a class attributes to be automatically
            tracked. e.g. ``self.fc = nn.Linear(3, 2)``. 
            
        Args:
            config (dict): a dictionary of configurations. 
        """
        pass
    
    @abstractmethod
    def prepare(self, config, **kwargs):
        r"""Prepare additional things for the agent. 
        
        For example, one could define optimizer and learning rate scheduler here. 
        
        Args:
            config (dict): a dictionary of configurations. 
            **kwargs: keyword arguments to specify the preparation. 
        """
        pass
    
    @abstractmethod
    def reset(self, config, **kwargs):
        r"""Reset the agent. 
        
        For example, this can be used for resetting the hidden state for recurrent neural networks. 
        
        Args:
            config (dict): a dictionary of configurations. 
            **kwargs: keyword arguments to specify reset function. 
        """
        pass
        
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
    
    @property
    def env_spec(self):
        r"""Returns the environment specifications. """
        return self._env_spec
        
    @property
    def observation_space(self):
        r"""Returns the observation space that policy performs on. """
        return self.env_spec.observation_space
    
    @property
    def action_space(self):
        r"""Returns the action space that policy performs on. """
        return self.env_spec.action_space
    
    @property
    @abstractmethod
    def recurrent(self):
        r"""Returns whether the agent is recurrent. """
        pass
    
    def __repr__(self):
        r"""Returns a string representation of the agent. """
        string = self.__class__.__name__
        string += f'\n\tEnvSpec: {self.env_spec}'
        string += f'\n\tRecurrent: {self.recurrent}'
        
        return string
