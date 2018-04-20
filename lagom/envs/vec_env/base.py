class VecEnv(object):
    """
    An asynchronous, vectorized environment. 
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
       
    def step_async(self, actions):
        """
        All environments take a step with the given actions. 
        
        Call step_wait() to obtain the outputs. 
        
        Note: Do not call this function when a step_async is pending
        """
        raise NotImplementedError
        
    def step_wait(self):
        """
        Wait for step_async(). 
        
        Returns:
            observations (array): single or a tuple of observations
            rewards (array): rewards
            dones (array): booleans of episode terminations
            infos (array): info objects
        """
        raise NotImplementedError
        
    def step(self, actions):
        self.step_async(actions)
        
        return self.step_wait()
    
    def reset(self):
        """
        Reset all environments and returns a tuple of observations. 
        
        step_async() will be cancelled. 
        """
        raise NotImplementedError
        
    def render(self):
        pass
        
    def close(self):
        """
        Closing the environments
        """
        raise NotImplementedError