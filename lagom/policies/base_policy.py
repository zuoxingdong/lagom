from abc import ABC
from abc import abstractmethod

from lagom.core.networks import BaseRNN

from lagom.envs.vec_env import VecEnv


class BasePolicy(ABC):
    r"""Base class for all policies.
    
    Any policy should subclass this class.
    
    The subclass should implement at least the following:
    
    - :meth:`__call__`
    
    .. note::
        
        For the consistency of different variants of policies and fast prototyping, we restrict that all
        policies should deal with VecEnv (batched data).
    
    """
    def __init__(self, config, network, env_spec, device, **kwargs):
        r"""Initialize the policy. 
        
        Args:
            config (dict): A dictionary of configurations. 
            network (BaseNetwork): a neural network as function approximator in the policy. 
            env_spec (EnvSpec): environment specification. 
            device (device): a PyTorch device for this policy. 
            **kwargs: keyword arguments to specify the policy. 
        """
        self.config = config
        self.network = network
        self._env_spec = env_spec
        self.device = device
        
        msg = f'expected type VecEnv, got {type(self.env_spec.env)}'
        assert isinstance(self.env_spec.env, VecEnv), msg
        
        # Set all keyword arguments
        for key, val in kwargs.items():
            self.__setattr__(key, val)
        
    @abstractmethod
    def __call__(self, x, out_keys=['action'], info={}, **kwargs):
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
            info (dict): a dictionary of additional information useful for action selection e.g. mask RNN states
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
    def recurrent(self):
        r"""Returns whether the policy is recurrent. """
        if isinstance(self.network, BaseRNN):
            return True
        else:
            return False
    
    def reset_rnn_states(self, batch_size=None):
        r"""Reset the current RNN states. """
        if batch_size is None:
            batch_size = self.env_spec.num_env
            
        if self.recurrent:
            self.rnn_states = self.network.init_hidden_states(config=self.config, batch_size=batch_size)
        else:
            raise TypeError('the network must be BaseRNN type. ')
            
    def update_rnn_states(self, rnn_states):
        r"""Update the current RNN states. """
        if self.recurrent:
            self.rnn_states = rnn_states
        else:
            raise TypeError('the network must be BaseRNN type. ')
        
    def __repr__(self):
        r"""Returns a string representation of the policy network. """
        string = self.__class__.__name__
        string += f'\n\tNetwork: {self.network.__repr__()}'
        string += f'\n\tRecurrent: {self.recurrent}'
        
        return string
