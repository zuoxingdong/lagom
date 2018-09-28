from abc import ABC
from abc import abstractmethod


class Env(ABC):
    r"""Base class for all environment used in lagom. 
    
    .. note::
    
        The main use for this class is to wrap all externial RL environment with 
        OpenAI Gym-like environment API, though not limited to Gym. 
    
    The subclass should implement at least the following:

    - :meth:`step`
    - :meth:`reset`
    - :meth:`render`
    - :meth:`close`
    - :meth:`seed`
    - :meth:`T`
    - :meth:`observation_space`
    - :meth:`action_space`
    - :meth:`max_episode_reward`
    - :meth:`reward_range`
    
    """
    @abstractmethod
    def step(self, action):
        r"""Execute the given action for one time step through the environment's dynamics. 
        
        .. note::
        
            When the episode terminates (i.e. ``done=True``), one should call :meth:`reset`
            before call this method again. 
        
        Args:
            action (object): a given action to the environment
            
        Returns
        -------
        observation : object
            the current observation agent receives after executing the given action.
        reward : float
            a scalar reward signal
        done : bool
            if ``True``, then the episode terminates.
        info : dict
            a dictionary of additional information
            
        """
        pass
        
    @abstractmethod
    def reset(self):
        r"""Reset the state of the environment and return an initial observation.
        
        Returns
        -------
        observation : object
            initial observation after resetting the environment
        """
        pass
        
    @abstractmethod
    def render(self, mode='human'):
        r"""Render the environment. 
        
        Args:
            mode (str): the mode for the rendering. Two modes are supported:
                * 'human': render to the current display
                * 'rgb_array': numpy array with shape [x, y, 3] for RGB values.
        """
        pass
    
    @abstractmethod
    def close(self):
        r"""Close the environment. 
        
        .. note::
        
            This will be automatically called when garbage collected or program exited. 
        
        .. note::
        
            Override this method to do any further cleanup. 
            
        """
        pass
    
    @abstractmethod
    def seed(self, seed):
        r"""Set the random seed of the environment. 
        
        Args:
            seed (int): the seed to initialize the pseudo-random number generator. 
        """
        pass
        
    @property
    def unwrapped(self):
        r"""Unwrap this environment. 
        
        Useful for sequential wrappers applied, it can access information from the original environment. 
        """
        return self
    
    @property
    @abstractmethod
    def observation_space(self):
        r"""Returns a :class:`Space` object to define the observation space. """
        pass
        
    @property
    @abstractmethod
    def action_space(self):
        r"""Returns a :class:`Space` object to define the action space. """
        pass
    
    @property
    @abstractmethod
    def T(self):
        r"""Maximum horizon of the environment, if available. """
        pass
        
    @property
    @abstractmethod
    def max_episode_reward(self):
        r"""Maximum episodic rewards of the environment, if available. """
        pass
    
    @property
    @abstractmethod
    def reward_range(self):
        r"""Returns a tuple of min and max possible rewards. 
        
        .. note::
        
            By default, it could be infinite range i.e. ``(-float('inf'), float('inf'))``. 
            One can also set a narrower range. 
        
        """
        pass
