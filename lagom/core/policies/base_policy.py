from lagom.envs.vec_env import VecEnv


class BasePolicy(object):
    r"""Base class for all policies.
    
    Any policy should subclass this class.
    
    The subclass should implement at least the following:
    
    - :meth:`__call__`
    
    .. note::
        
        For the consistency of different variants of policies and fast prototyping, we restrict that all
        policies should deal with VecEnv (batched data).
    
    """
    def __init__(self, config, network, env_spec, **kwargs):
        r"""Initialize the policy. 
        
        Args:
            config (dict): A dictionary of configurations. 
            network (BaseNetwork): a neural network as function approximator in the policy. 
            env_spec (EnvSpec): environment specification. 
            **kwargs: keyword arguments to specify the policy. 
        """
        self.config = config
        self.network = network
        self.env_spec = env_spec
        
        msg = f'expected type VecEnv, got {type(self.env_spec.env)}'
        assert isinstance(self.env_spec.env, VecEnv), msg
        
        # Set all keyword arguments
        for key, val in kwargs.items():
            self.__setattr__(key, val)
        
    def __call__(self, x):
        r"""Define the computation of the policy given input data at every call. 
        
        Should be overridden by all subclasses.
        
        Args:
            x (object): input data to the policy. 
            
        Returns
        -------
        out_policy : dict
            A dictionary of output data about the computation of the policy. It should contain
            at least one key 'action'. Other possible keys include ['action_logprob', 'state_value'
            'entropy', 'perplexity']. 
        """
        raise NotImplementedError
        
    def __repr__(self):
        r"""Returns a string representation of the policy network. """
        string = self.__class__.__name__ + '\n'
        string += '\tNetwork: ' + self.network.__repr__() + '\n'
        
        return string
