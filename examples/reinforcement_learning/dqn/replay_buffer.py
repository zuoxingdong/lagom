import numpy as np
import random

from .segment_tree import MinSegmentTree
from .segment_tree import SumSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
        r"""Constructor. 
        
        Args:
            size (int): max number of transitions to store. 
        """
        self.storage = []
        self.size = size
        self.next_idx = 0
        
    def __len__(self):
        return len(self.storage)
    
    def add(self, s, a, r, s_next, done):
        D = (s, a, r, s_next, done)
        if self.next_idx >= len(self.storage):
            self.storage.append(D)
        else:
            self.storage[self.next_idx] = D
        self.next_idx = (self.next_idx + 1)%self.size
        
    def _encode_sample(self, list_idx):
        all_s, all_a, all_r, all_s_next, all_done = [], [], [], [], []
        for idx in list_idx:
            s, a, r, s_next, done = self.storage[idx]
            all_s.append(np.array(s, copy=False))
            all_a.append(np.array(a, copy=False))
            all_r.append(r)
            all_s_next.append(np.array(s_next, copy=False))
            all_done.append(done)
        return np.array(all_s), np.array(all_a), np.array(all_r), np.array(all_s_next), np.array(all_done)
    
    def sample(self, batch_size):
        r"""Sample a batch of experiences. 
        
        Args:
            batch_size (int): number of transitions to sample
            
        Returns
        -------
        batch_s : array
            batch of observations
        batch_a : array
            batch of actions
        batch_r : array
            batch of rewards
        batch_s_next : array
            batch of next observations
        batch_done : array
            batch of dones, 1: True, 0: False
        """
        list_idx = [random.randint(0, len(self.storage)) for _ in range(batch_size)]
        return self._encode_sample(list_idx)
    
    
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        r"""Define prioritized experience reply. 
        
        Args:
            size (int): max number of transitions to store. 
            alpha (float): how much prioritization, 0: no prioritization, 1: full prioritization
        """
        super().__init__(size)
        assert alpha >= 0
        self.alpha = alpha
        
        capacity = 1
        while capacity < size:
            capacity *= 2
            
        self.sum = SumSegmentTree(capacity)
        self.min = MinSegmentTree(capacity)
        self.max_priority = 1.0
        
    def add(self, *args, **kwargs):
        idx = self.next_idx
        super().add(*args, **kwargs)
        self.sum[idx] = self.max_priority**self.alpha
        self.min[idx] = self.max_priority**self.alpha
        
    def _sample_proportional(self, batch_size):
        out = []
        p_total = self.sum.sum(0, len(self.storage) - 1)
        every_range_len = p_total/batch_size
        for i in range(batch_size):
            mass = random.random()*every_range_len + i*every_range_len
            idx = self.sum.find_prefixsum_idx(mass)
            out.append(idx)
        return out
    
    def sample(self, batch_size, beta):
        assert beta > 0
        
        list_idx = self._sample_proportional(batch_size)
        
        weights = []
        p_min = self.min.min()/self.sum.sum()
        max_weight = (p_min*len(self.storage))**(-beta)
        
        for idx in list_idx:
            p_sample = self.sum[idx]/self.sum.sum()
            weight = (p_sample*len(self.storage))**(-beta)
            weights.append(weight/max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(list_idx)
        return tuple(list(encoded_sample) + [weights, list_idx])
    
    def update_priorities(self, list_idx, priorities):
        assert len(list_idx) == len(priorities)
        for idx, priority in zip(list_idx, priorities):
            assert priority > 0
            assert idx >= 0 and idx < len(self.storage)
            self.sum[idx] = priority**self.alpha
            self.min[idx] = priority**self.alpha
            
            self.max_priority = max(self.max_priority, priority)
