from itertools import chain

import numpy as np


class BatchHistory(object):
    def __init__(self, env):
        self.env = env
        self.n = self.env.num_env
        
        self.s = [[[]] for _ in range(self.n)]
        self.a = [[[]] for _ in range(self.n)]
        self.r = [[[]] for _ in range(self.n)]
        self.done = [[[]] for _ in range(self.n)]
        self.info = [[[]] for _ in range(self.n)]
        
        self.batch_info = []
        
        self.stops = [False for _ in range(self.n)]
    
    def stop(self, n):
        self.stops[n] = True
    
    def add(self, observations, actions, rewards, dones, infos, batch_info):
        assert all(len(item) == self.n for item in [observations, actions, rewards, dones, infos])
        assert not all(self.stops)
        
        assert isinstance(batch_info, dict)
        self.batch_info.append(batch_info)
    
        for n in range(self.n):
            if not self.stops[n]:
                # create new sub-list if done=True
                if len(self.done[n][-1]) > 0 and self.done[n][-1][-1]:
                    self.s[n].append([])
                    self.a[n].append([])
                    self.r[n].append([])
                    self.done[n].append([])
                    self.info[n].append([])
                
                self.s[n][-1].append(observations[n])
                self.a[n][-1].append(actions[n])
                self.r[n][-1].append(rewards[n])
                self.done[n][-1].append(dones[n])
                self.info[n][-1].append(infos[n])    
    
    @property
    def num_traj(self):
        return [len(self.done[n]) for n in range(self.n)]
    
    @property
    def T(self):
        return len(self.batch_info)
    
    @property
    def Ts(self):
        return [[len(x) for x in done] for done in self.done]
    
    @property
    def Ts_flat(self):
        out = []
        for T in self.Ts:
            out += T
        return out
    
    @property
    def observations(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype
        out = np.zeros((sum(self.num_traj), max(self.Ts_flat)) + shape, dtype=dtype)
        counter = 0
        for s in self.s:
            for obs in s:
                out[counter, :len(obs), ...] = obs
                counter += 1
        assert counter == sum(self.num_traj)
        return out
    
    @property
    def batch_observations(self):
        out = []
        for s in self.s:
            out.append(np.stack(list(chain.from_iterable(s))))
        out = np.stack(out)
        return out
    
    @property
    def last_observations(self):
        shape = self.env.observation_space.shape
        dtype = self.env.observation_space.dtype
        out = np.zeros((sum(self.num_traj),) + shape, dtype=dtype)
        counter = 0
        for n in range(self.n):
            for m in range(len(self.s[n])):
                if self.done[n][m][-1]:
                    out[counter, ...] = self.info[n][m][-1]['terminal_observation']
                else:
                    out[counter, ...] = self.s[n][m][-1]
                counter += 1
        assert counter == sum(self.num_traj)
        return out
        
    @property
    def actions(self):
        shape = self.env.action_space.shape
        dtype = self.env.action_space.dtype
        out = np.zeros((sum(self.num_traj),) + shape, dtype=dtype)
        counter = 0
        for a in self.a:
            for action in a:
                out[counter, :len(action), ...] = action
                counter += 1
        assert counter == sum(self.num_traj)
        return out
    
    @property
    def batch_actions(self):
        out = []
        for a in self.a:
            out.append(np.stack(list(chain.from_iterable(a))))
        out = np.stack(out)
        return out
        
    @property
    def rewards(self):
        out = np.zeros((sum(self.num_traj), max(self.Ts_flat)), dtype=np.float32)
        counter = 0
        for r in self.r:
            for reward in r:
                out[counter, :len(reward)] = reward
                counter += 1
        assert counter == sum(self.num_traj)
        return out
    
    @property
    def batch_rewards(self):
        out = []
        for r in self.r:
            out.append(np.stack(list(chain.from_iterable(r))))
        out = np.stack(out)
        return out
    
    @property
    def dones(self):
        out = np.full((sum(self.num_traj), max(self.Ts_flat)), True, dtype=np.bool)
        counter = 0
        for done in self.done:
            for d in done:
                out[counter, :len(d)] = d
                counter += 1
        assert counter == sum(self.num_traj)
        return out
    
    @property
    def batch_dones(self):
        out = []
        for done in self.done:
            out.append(np.stack(list(chain.from_iterable(done))))
        out = np.stack(out)
        return out
    
    @property
    def masks(self):
        return np.logical_not(self.dones).astype(np.float32)
    
    @property
    def batch_masks(self):
        return np.logical_not(self.batch_dones).astype(np.float32)
    
    @property
    def validity_masks(self):
        out = np.zeros((sum(self.num_traj), max(self.Ts_flat)), dtype=np.float32)
        for n, T in enumerate(self.Ts_flat):
            out[n, :T] = 1.0
        return out
    
    @property
    def batch_validity_masks(self):
        out = np.zeros((self.n, self.T), dtype=np.float32)
        for n, T in enumerate(self.Ts):
            out[n, :sum(T)] = 1.0
        return out
    
    @property
    def infos(self):
        return self.info
    
    def get_batch_info(self, key):
        return [batch_info[key] for batch_info in self.batch_info]
    
    def __repr__(self):
        return f'{self.__class__.__name__}(N={self.n}, T={self.T}, num_traj={sum(self.num_traj)}, maxT={max(self.Ts_flat)})'
