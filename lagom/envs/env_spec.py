import numpy as np

from .env import Env
from .vec_env import VecEnv

from .spaces import Discrete
from .spaces import Box


class EnvSpec(object):
    r"""Summarize the specifications of the environment. 
    
    It collects useful properties of an environment which can be very convenient for
    designing generic APIs to train RL agents, such as observation and action spaces, 
    maximum allowed horizon, discrete or continuous control type etc. 
    """
    def __init__(self, env):
        r"""Initialize the environment specification. 
        
        Args:
            env (Env/VecEnv): an environment object. 
        """
        assert isinstance(env, (Env, VecEnv)), f'expected Env or VecEnv, got {type(env)}'
        
        self.env = env
    
    @property
    def observation_space(self):
        r"""Returns the observation space of the environment. """
        return self.env.observation_space
    
    @property
    def action_space(self):
        r"""Returns the action space of the environment. """
        return self.env.action_space
    
    @property
    def T(self):
        r"""Returns the maximum horizon of the environment. """
        return self.env.T
    
    @property
    def max_episode_reward(self):
        r"""Returns the maximum episodic rewards of the environment. """
        return self.env.max_episode_reward
    
    @property
    def reward_range(self):
        r"""Returns a tuple of min and max possible rewards. """
        return self.env.reward_range
    
    @property
    def control_type(self):
        r"""Returns a string to indicate if the environment is discrete or continuous control. 
        
        It returns one of the following strings:
        
        * 'Discrete': if the action space is of type :class:`Discrete`
        
        * 'Continuous': if the action space is of type :class:`Box`
        
        """
        if isinstance(self.env.action_space, Discrete):
            return 'Discrete'
        elif isinstance(self.env.action_space, Box):
            return 'Continuous'
        else:
            raise TypeError(f'expected Discrete or Box, got {type(self.env.action_space)}.')
            
    @property
    def is_vec_env(self):
        if isinstance(self.env, VecEnv):
            return True
        else:
            return False
            
    @property
    def num_env(self):
        r"""Returns the number of sub-environments for vectorized environment. """
        if isinstance(self.env, VecEnv):
            return self.env.num_env
        else:
            raise TypeError('the environment must be VecEnv')

    def __repr__(self):
        string = f'<{type(self).__name__}, {self.env}>\n'
        if isinstance(self.env, VecEnv):
            string += f'\tNumber of environments: {self.num_env}\n'
        string += f'\tObservation space: {self.observation_space}\n'
        string += f'\tAction space: {self.action_space}\n'
        string += f'\tControl type: {self.control_type}\n'
        string += f'\tT: {self.T}\n'
        string += f'\tMax episode reward: {self.max_episode_reward}\n'
        string += f'\tReward range: {self.reward_range}\n'
        string += f'\tIs VecEnv: {self.is_vec_env}'
        
        return string
