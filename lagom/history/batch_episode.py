import numpy as np

from .base_history import BaseHistory


class BatchEpisode(BaseHistory):
    def __init__(self, env_spec):
        super().__init__(env_spec)
        
        self.obs = [[] for _ in range(self.N)]
        self.a = [[] for _ in range(self.N)]
        self.r = [[] for _ in range(self.N)]
        self.done = [[] for _ in range(self.N)]
        self.info = [[] for _ in range(self.N)]
        
        self.batch_info = []
        
        self.completed = [False for _ in range(self.N)]
    
    def _add_with_complete(self, x, D):
        for n, (x_n, D_n) in enumerate(zip(x, D)):
            if not self.completed[n]:
                D_n.append(x_n)
    
    def add_observation(self, observations):
        self._check_len(observations)
        self._add_with_complete(observations, self.obs)
    
    def add_action(self, actions):
        self._check_len(actions)
        self._add_with_complete(actions, self.a)
        
    def add_reward(self, rewards):
        self._check_len(rewards)
        self._add_with_complete(rewards, self.r)
        
    def add_done(self, dones):
        self._check_len(dones)
        self._add_with_complete(dones, self.done)
        
    def add_info(self, infos):
        self._check_len(infos)
        self._add_with_complete(infos, self.info)
        
    def add_batch_info(self, info):
        assert isinstance(info, dict)
        self.batch_info.append(info)
        
    def set_completed(self, n):
        self.completed[n] = True
        
    @property
    def observations(self):
        return self.obs
    
    @property
    def numpy_observations(self):
        out = np.zeros((self.N, self.maxT+1) + self.env_spec.observation_space.shape, dtype=np.float32)  # plus 1: initial observation
        for n in range(self.N):
            obs = np.stack(self.obs[n], axis=0)
            t = obs.shape[0]
            out[n, :t, ...] = obs
            
        return out

    @property
    def actions(self):
        return self.a
    
    @property
    def numpy_actions(self):
        if self.env_spec.control_type == 'Discrete':
            out_shape = (self.N, self.maxT)
            out_dtype = np.int32
        elif self.env_spec.control_type == 'Continuous':
            out_shape = (self.N, self.maxT) + self.env_spec.action_space.shape
            out_dtype = np.float32
        out = np.zeros(out_shape, dtype=out_dtype)
        for n in range(self.N):
            action = np.stack(self.a[n], axis=0)
            t = action.shape[0]
            out[n, :t, ...] = action
            
        return out
    
    @property
    def rewards(self):
        return self.r
    
    @property
    def numpy_rewards(self):
        out = np.zeros((self.N, self.maxT), dtype=np.float32)
        for n in range(self.N):
            reward = np.array(self.r[n], dtype=np.float32)
            t = reward.shape[0]
            out[n, :t] = reward
            
        return out
    
    @property
    def dones(self):
        return self.done
    
    @property
    def numpy_dones(self):
        out = np.full((self.N, self.maxT), True, dtype=np.bool)
        for n in range(self.N):
            t = len(self.done[n])
            out[n, :t] = self.done[n]
        
        return out
    
    @property
    def masks(self):
        return [np.logical_not(d).astype(np.float32) for d in self.done]
    
    @property
    def infos(self):
        return self.info
    
    @property
    def batch_infos(self):
        return self.batch_info
    
    @property
    def completes(self):
        return self.completed
    
    @property
    def Ts(self):
        return np.array([len(d) for d in self.done])
    
    @property
    def maxT(self):
        return np.max(self.Ts)
    
    @property
    def total_T(self):
        return sum(self.Ts)
    
    def __repr__(self):
        string = f'{self.__class__.__name__}(N={self.N}, maxT={(self.maxT)})'
        return string
