from lagom.envs import Env


class Wrapper(Env):
    r"""Wraps the environment to allow a modular transformation. 
    
    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code. 
    
    .. note::
    
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    
    """
    def __init__(self, env):
        assert isinstance(env, Env), f'expected Env type, got {type(env)}'
            
        self.env = env
        
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        return self.env.reset()
    
    def render(self, mode='human', **kwargs):
        # kwargs supports Mujoco render to specify width and height
        return self.env.render(mode, **kwargs)
    
    def close(self):
        return self.env.close()
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)
    
    @property
    def unwrapped(self):
        return self.env.unwrapped
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def T(self):
        return self.env.T
    
    @property
    def max_episode_reward(self):
        return self.env.max_episode_reward
    
    @property
    def reward_range(self):
        return self.env.reward_range
    
    def __repr__(self):
        return f'<{type(self).__name__}, {self.env}>'
    

class ObservationWrapper(Wrapper):
    r"""Observation wrapper modifies the received observation after each call of :meth:`step` and :meth:`reset`. 
    
    It is the base class for all observation wrappers. 
    
    .. note::
    
        When observation is modified, the observation space should also be adjusted accordingly.
    
    The subclass should implement at least the following:
    
    - :meth:`process_observation`
    - :meth:`observation_space`
    
    """
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        return self.process_observation(observation), reward, done, info
    
    def reset(self):
        observation = self.env.reset()
        
        return self.process_observation(observation)
    
    def process_observation(self, observation):
        r"""Process the observation. 
        
        Args:
            observation (object): original observation. 
            
        Returns
        -------
        out : object
            processed observation
        """
        raise NotImplementedError
        
    @property
    def observation_space(self):
        r"""Adjusted observation space according to :meth:`process_observation`. """
        raise NotImplementedError
        
        
class ActionWrapper(Wrapper):
    r"""Action wrapper modifies the action before calling :meth:`step`. 
    
    It is the base class for all action wrappers.
    
    .. note::
    
        When action is modified, the action space should also be adjusted accordingly. 
    
    The subclass should implement at least the following:
    
    - :meth:`process_action`
    - :meth:`action_space`
    
    """
    def step(self, action):
        return self.env.step(self.process_action(action))
    
    def process_action(self, action):
        r"""Process the action. 
        
        Args:
            action (object): original action. 
            
        Returns
        -------
        out : object
            processed action
        """
        raise NotImplementedError
        
    @property
    def action_space(self):
        r"""Adjusted action space according to :meth:`process_action`. """
        raise NotImplementedError
        
        
class RewardWrapper(Wrapper):
    r"""Reward wrapper modifies the received reward after each function call of :meth:`step`. 
    
    It is the base class for all reward wrappers. 
    
    .. note::
    
        When reward is modified, it is recommended to adapt max_episode_reward and reward_range.
        But it is not strictly forced to do so. 
    
    The subclass should implement at least the following:
    
    - :meth:`process_reward`
    
    """
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        return observation, self.process_reward(reward), done, info
    
    def process_reward(self, reward):
        r"""Process the reward. 
        
        Args:
            reward (float): original reward
        
        Returns
        -------
        out : float
            processed reward
        """
        raise NotImplementedError
