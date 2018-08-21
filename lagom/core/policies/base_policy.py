class BasePolicy(object):
    """
    Base class for the policy. 
    
    It receives user-defined network (must be of types in lagom.core.networks),
    environment specification (of type EnvSpec) and configuration. 
    
    All inherited subclasses should at least implement the following functions
    1. __call__(self, x)
    2. process_network_output(self, network_out)
    """
    def __init__(self, network, env_spec, config, **kwargs):
        """
        Args:
            network (BaseNetwork): an instantiated user-defined network. 
            env_spec (EnvSpec): environment specification. 
            config (dict): A dictionary for the configuration. 
            *kwargs: keyword aguments used to specify the policy. 
        """
        self.network = network
        self.env_spec = env_spec
        self.config = config
        
        # Set all keyword arguments
        for key, val in kwargs.items():
            self.__setattr__(key, val)
        
    def __call__(self, x):
        """
        User-defined function to run the policy network given the input. 
        
        It should use the forward() function of internal network, e.g. `self.network(x)`
        
        Note that it must return a dictionary of output. 
        
        Args:
            x (Tensor): input data. 
            
        Returns:
            network_out (dict): A dictionary of output data from running the policy network of given input. 
                It must contain at least one key, 'action'. Other possible keys include 
                ['action_logprob', 'state_value']
        """
        raise NotImplementedError
        
    def process_network_output(self, network_out):
        """
        User-defined function to support additional processing of the 
        output from the internal network. 
        
        It can also return a processed output. If there is nothing to do, 
        then return it back, i.e. return network_out
        
        Args:
            network_out (dict): Dictionary of data returned from forward pass of internal network. 
            
        Returns:
            processed_network_out (dict): A dictionary of processed network output. 
                It will be returned together in __call__. Default to return back
                of network_out
        """
        raise NotImplementedError
