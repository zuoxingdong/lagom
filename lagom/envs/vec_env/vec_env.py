from abc import ABC
from abc import abstractmethod


class VecEnv(ABC):
    """
    An abstract class for asynchronous, vectorized environment. 
    
    Note that all vectorized environments should share the same observation and action spaces.
    
    All inherited subclasses should at least implement the following functions:
    1. step_asyn(self, actions)
    2. step_wait(self)
    3. reset(self)
    4. render(self, mode='human')
    5. close(self)
    6. seed(self, seeds)
    7. @property: T(self)
    """
    def __init__(self, list_make_env, observation_space, action_space):
        self.list_make_env = list_make_env
        self.num_env = len(self.list_make_env)
        self.observation_space = observation_space
        self.action_space = action_space
        
    @abstractmethod
    def step_async(self, actions):
        """
        Notify all the environments to execute the given actions, each for one environment. 
        
        Call step_wait() to receive all the results. 
        
        Note that do not call this function if it is already pending. 
        
        Args:
            actions (list): a list of given actions, each for one environment. 
        """
        pass
        
    @abstractmethod
    def step_wait(self):
        """
        Wait for the job in step_async() to finish and return all the results. 
        
        Returns:
            observations (list of object): A list of current observations that agent receives from all 
                environments after executing the given actions. 
            rewards (list of float): A list of scalar rewards from all environments. 
            dones (list of bool): A list booleans from all environments. If True, then the episode terminates.
            infos (list of dict): A list of debugging informations from all environments.
        """
        pass
        
    def step(self, actions):
        """
        Execute all the given actions, each for one environment, for one time step through
        the environments' dynamics. 
        
        Args:
            actions (list): a list of given actions, each for one environment. 
            
        Returns:
            observations (list of object): A list of current observations that agent receives from all 
                environments after executing the given actions. 
            rewards (list of float): A list of scalar rewards from all environments. 
            dones (list of bool): A list booleans from all environments. If True, then the episode terminates.
            infos (list of dict): A list of debugging informations from all environments.
        """
        # Execute the actions in all environments asynchronously
        self.step_async(actions)
        
        # wait to receive the results and return them
        return self.step_wait()
    
    @abstractmethod
    def reset(self):
        """
        Reset the states of all the environments and return a list of initial observations.
        
        Note that the step_async() will be cancelled if it is still woring. 
        
        Returns:
            observations (list): The list of initial observations from all environments. 
        """
        pass
    
    @abstractmethod
    def render(self, mode='human'):
        """
        Render the vectorized environment. 
        
        Args:
            mode (str): The mode for the rendering. Two modes are supported.
                        - 'human': Often pop up a rendered window
                        - 'rgb_array': numpy array with shape [x, y, 3] for RGB values.
        """
        pass
    
    @abstractmethod
    def close(self):
        """
        This will be automatically called when garbage collected or program exited. 
        
        Override this method to do any further cleanup. 
        """
        pass
    
    @abstractmethod
    def seed(self, seeds):
        """
        Set the random seeds for each of the environment. 
        
        Args:
            seeds (list): List of seeds to initialize the pseudo-random number generator
                for all environments. 
        """
        # TODO: make it better to use
        pass
    
    @property
    def unwrapped(self):
        """
        Unwrap the vectorized environment. 
        
        Useful for sequential wrappers applied, it can access information from those environments.
        """
        return self
    
    @property
    @abstractmethod
    def T(self):
        """
        Horizon of the environment, if available
        """
        pass
