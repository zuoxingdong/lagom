from abc import ABC
from abc import abstractmethod

import numpy as np


class BaseHistory(ABC):
    r"""Base class for all history of interactions with environment. 
    
    It stores observations, actions, rewards, dones, infos etc. for each time step. 
    """
    def __init__(self, env_spec):
        self.env_spec = env_spec
        
    def _check_len(self, x):
        assert len(x) == self.N
    
    @abstractmethod
    def add_observation(self, observations):
        pass
    
    @abstractmethod
    def add_action(self, actions):
        pass
    
    @abstractmethod
    def add_reward(self, rewards):
        pass
    
    @abstractmethod
    def add_done(self, dones):
        pass
    
    @abstractmethod
    def add_info(self, infos):
        pass
    
    @abstractmethod
    def add_batch_info(self, info):
        pass
    
    @property
    @abstractmethod
    def numpy_observations(self):
        pass
    
    @property
    @abstractmethod
    def numpy_actions(self):
        pass
        
    @property
    @abstractmethod
    def numpy_rewards(self):
        pass
        
    @property
    @abstractmethod
    def numpy_dones(self):
        pass
        
    @property
    def numpy_masks(self):
        return np.logical_not(self.numpy_dones).astype(np.float32)
    
    @property
    @abstractmethod
    def infos(self):
        pass
    
    @property
    @abstractmethod
    def batch_infos(self):
        pass
    
    @property
    def N(self):
        return self.env_spec.num_env
    
    @property
    @abstractmethod
    def total_T(self):
        pass
