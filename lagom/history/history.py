import numpy as np


class History(object):
    def __init__(self, env_spec, T):
        self.env_spec = env_spec
        self._T = T
        
        obs_shape = self.env_spec.observation_space.shape
        # T + 1: plus initial observation
        self._observations = np.zeros((self.N, self.T+1) + obs_shape, dtype=np.float32)
        self._rewards = np.zeros((self.N, self.T), dtype=np.float32)
        self._dones = np.ones((self.N, self.T), dtype=np.bool)
        self._infos = [None]*self.T
        
        self.extra_info = {}
        
    def add(self, name, value):
        assert not hasattr(self, name)
        setattr(self, name, value)
        
    def add_t(self, name, t, value):
        if not hasattr(self, name):
            setattr(self, name, [None]*self.T)
        
        getattr(self, name)[t] = value
        
    def get(self, name):
        return getattr(self, name)
        
    def get_t(self, name, t):
        return getattr(self, name)[t]
    
    def add_extra_info(self, key, value):
        self.extra_info[key] = value
        
    def get_extra_info(self, key):
        return self.extra_info[key]
        
    @property
    def observations(self):
        return self._observations
    
    @observations.setter
    def observations(self, obs):
        self._observations = np.asarray(obs).astype(np.float32)
        
    @property
    def rewards(self):
        return self._rewards
    
    @rewards.setter
    def rewards(self, r):
        self._rewards = np.asarray(r).astype(np.float32)
        
    @property
    def dones(self):
        return self._dones
    
    @dones.setter
    def dones(self, done):
        self._dones = np.asarray(done).astype(np.bool)
        
    @property
    def infos(self):
        return self._infos
    
    @infos.setter
    def infos(self, info):
        self._infos = info
        
    @property
    def N(self):
        return self.env_spec.num_env
        
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, new_T):
        self._T = new_T
    
    @property
    def masks(self):
        return np.logical_not(self.dones).astype(np.int32)
    
    def __repr__(self):
        string = f'{self.__class__.__name__}: N={self.N}, T={self.T}'
        
        return string
