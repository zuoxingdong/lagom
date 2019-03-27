import numpy as np


class Trajectory(object):
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        
        self.completed = False
        
    def add_observation(self, observation):
        assert not self.completed
        self.observations.append(observation)
    
    @property
    def numpy_observations(self):
        out = np.concatenate(np.asarray(self.observations), axis=0)
        assert out.shape[0] == len(self) + 1  # plus initial observation
        return out
    
    @property
    def last_observation(self):
        return self.observations[-1]
    
    @property
    def reach_terminal(self):
        return self.dones[-1] and 'TimeLimit.truncated' not in self.infos[-1]
    
    def add_action(self, action):
        assert not self.completed
        self.actions.append(action)
        
    @property
    def numpy_actions(self):
        return np.concatenate(np.asarray(self.actions), axis=0)
        
    def add_reward(self, reward):
        assert not self.completed
        self.rewards.append(reward)
        
    @property
    def numpy_rewards(self):
        return np.asarray(self.rewards)
    
    def add_done(self, done):
        assert not self.completed
        self.dones.append(done)
        if done:
            self.completed = True
        
    @property
    def numpy_dones(self):
        return np.asarray(self.dones)
    
    @property
    def numpy_masks(self):
        return 1. - self.numpy_dones
        
    def add_info(self, info):
        assert not self.completed
        self.infos.append(info)
    
    def get_all_info(self, key):
        return [info[key] for info in self.infos]
    
    def __len__(self):
        return len(self.dones)
        
    def __repr__(self):
        return f'Trajectory({len(self)})'
