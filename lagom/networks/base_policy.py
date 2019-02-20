from abc import ABC
from abc import abstractmethod

from lagom.networks import Module
from lagom.networks import BaseRNN

from lagom.envs.vec_env import VecEnv


class BasePolicy(Module, ABC):
    r"""Base class for all policies.
    
    Any policy should subclass this class.
    
    .. note::
        
        For the consistency of different variants of policies and fast prototyping, we restrict that all
        policies should deal with VecEnv (batched data).
    
    """
    def __init__(self, config, env_spec, device, **kwargs):
        r"""Initialize the policy. 
        
        Args:
            config (dict): A dictionary of configurations. 
            env_spec (EnvSpec): environment specification. 
            device (device): a PyTorch device for this policy. 
            **kwargs: keyword arguments to specify the policy. 
        """
        super(Module, self).__init__(**kwargs)
        
        self.config = config
        self._env_spec = env_spec
        self.device = device
        
        msg = f'expected type VecEnv, got {type(self.env_spec.env)}'
        assert isinstance(self.env_spec.env, VecEnv), msg
        
        self.make_networks(self.config)
        
        self.make_optimizer(self.config, **kwargs)
    
    @abstractmethod
    def make_networks(self, config):
        r"""Create all network modules for the policy.
        
        For example, this allows to easily create separate value network or probabilistic action head. 
        
        .. note::
        
            All created networks must be assigned as a class attributes to be automatically
            tracked. e.g. ``self.fc = nn.Linear(3, 2)``. 
        
        Args:
            config (dict): a dictionary of configurations. 
        """
        pass
    
    @abstractmethod
    def make_optimizer(self, config, **kwargs):
        r"""Create optimization related objects e.g. optimizer, learning rate scheduler.
        
        Args:
            config (dict): a dictionary of configurations. 
            **kwargs: keyword arguments for more specifications. 
        """
        pass
    
    @abstractmethod
    def optimizer_step(self, config, **kwargs):
        r"""Define one gradient step to the optimizer. 
        
        It is also posssible to include learning rate scheduling, gradient clipping etc.
        
        Args:
            config (dict): a dictionary of configurations. 
            **kwargs: keyword arguments for more specifications. 
        """
        pass
    
    @abstractmethod
    def reset(self, config, **kwargs):
        r"""Reset the policy.
        
        For example, this can be used for resetting the hidden state for recurrent neural networks. 
        
        Args:
            config (dict): a dictionary of configurations. 
            **kwargs: keyword arguments to specify reset function. 
        """
        pass

    @abstractmethod
    def __call__(self, x, out_keys=['action'], **kwargs):
        r"""Define the computation of the policy given input data at every call. 
        
        Should be overridden by all subclasses.
        
        .. note::
        
            There is an option to select metrics for the policy to calculate and only selected items will
            be calculated and returned e.g. ``out_keys=['action', 'action_logprob', 'entropy']``. 
            This is very useful to dramatically speedup in some scenarios. For example in ES, it
            turns out that outputing all metrics of an action distribution makes training extremly
            slow and only action is useful but others like log-probability, entropy etc. 
        
        Args:
            x (object): input data to the policy. 
            out_keys (list, optional): a list of required metrics for the policy to output. 
                Default: ``['action']``
            **kwargs: keyword aguments used to specify the policy execution.
            
        Returns
        -------
        out_policy : dict
            A dictionary of output data about the computation of the policy. It should contain
            at least one key 'action'. Other possible keys include ['action_logprob', 'state_value'
            'entropy', 'perplexity']. 
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
        r"""Returns whether the policy is recurrent. """
        pass
        
    def __repr__(self):
        r"""Returns a string representation of the policy network. """
        string = self.__class__.__name__
        string += f'\n\tEnvSpec: {self.env_spec}'
        string += f'\n\tRecurrent: {self.recurrent}'
        
        return string
