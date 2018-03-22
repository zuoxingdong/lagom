class Env(object):
    """Base class for environment"""
    def step(self, action):
        """
        Execute the given action for one time step through the environment's dynamics. 
        
        Args:
            action (object): a given action to the environment
            
        Returns:
            observation (object): The current observation agent receives after executing the given action.
            reward (float): A scalar reward signal
            done (bool): True if the episode terminates.
            info (dict): Debugging information.
        """
        raise NotImplementedError
        
    def reset(self):
        """
        Reset the state of the environment and return an initial observation.
        
        Returns:
            observation (object): The observation agents receives for the initial state.
        """
        raise NotImplementedError
        
    def render(self, mode='human'):
        """
        Render the environment. 
        
        Args:
            mode (str): The mode for the rendering. Two modes are supported.
                        - 'human': Often pop up a rendered window
                        - 'rgb_array': numpy array with shape [x, y, 3] for RGB values.
        """
        raise NotImplementedError
        
    def seed(self, seed):
        """
        Set the random seed of the environment. 
        
        Args:
            seed (int): The seed to initialize the pseudo-random number generator. 
        """
        raise NotImplementedError
        
    def clean(self):
        """
        Clean up.
        """
        raise NotImplementedError
        
    def get_source_env(self):
        """
        Return the original environment object. 
        Useful for sequential wrappers, to access the information from original environment. 
        """
        raise NotImplementedError
        
    @property
    def T(self):
        """
        Horizon of the environment, if available
        """
        raise NotImplementedError
        
    @property
    def observation_space(self):
        """
        Return a Space object to define the observation space.
        """
        raise NotImplementedError
        
    @property
    def action_space(self):
        """
        Return a Space object to define the action space.
        """
        raise NotImplementedError