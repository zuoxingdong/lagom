import numpy as np

from .base_history import BaseHistory


class BatchSegment(BaseHistory):
    def __init__(self, env_spec, T):
        super().__init__(env_spec)
        
        self._T = T
        
        if self.env_spec.control_type == 'Discrete':
            action_shape = ()
            action_dtype = np.int32
        elif self.env_spec.control_type == 'Continuous':
            action_shape = self.env_spec.action_space.shape
            action_dtype = np.float32
        
        self.obs = np.zeros((self.N, self.T+1) + self.env_spec.observation_space.shape, dtype=np.float32)  # plus 1: initial observation
        self.a = np.zeros((self.N, self.T) + action_shape, dtype=action_dtype)
        self.r = np.zeros((self.N, self.T), dtype=np.float32)
        self.done = np.full((self.N, self.T), True, dtype=np.bool)
        self.info = [[] for _ in range(self.N)]
        
        self.batch_info = []
        
    def add_observation(self, t, observations):
        self._check_len(observations)
        self.obs[:, t, ...] = observations
        
    def add_action(self, t, actions):
        self._check_len(actions)
        self.a[:, t, ...] = actions
        
    def add_reward(self, t, rewards):
        self._check_len(rewards)
        self.r[:, t] = rewards
        
    def add_done(self, t, dones):
        self._check_len(dones)
        self.done[:, t] = dones
        
    def add_info(self, infos):
        self._check_len(infos)
        for n, info in enumerate(infos):
            self.info[n].append(info)
        
    def add_batch_info(self, info):
        assert isinstance(info, dict)
        self.batch_info.append(info)
        
    @property
    def numpy_observations(self):
        return self.obs
    
    @property
    def numpy_actions(self):
        return self.a
    
    @property
    def numpy_rewards(self):
        return self.r
    
    @property
    def numpy_dones(self):
        return self.done
    
    @property
    def infos(self):
        return self.info
    
    @property
    def batch_infos(self):
        return self.batch_info
    
    @property
    def T(self):
        return self._T
    
    @property
    def total_T(self):
        return self.N*self.T
    
    def __repr__(self):
        string = f'{self.__class__.__name__}(N={self.N}, T={(self.T)})'
        return string
