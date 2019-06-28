import numpy as np


class Trajectory(object):
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.step_infos = []
        
    def __len__(self):
        return len(self.step_infos)
        
    @property
    def completed(self):
        return len(self.step_infos) > 0 and self.step_infos[-1].last
    
    @property
    def reach_time_limit(self):
        return self.step_infos[-1].time_limit
    
    @property
    def reach_terminal(self):
        return self.step_infos[-1].terminal
        
    def add_observation(self, observation):
        assert not self.completed
        self.observations.append(observation)
    
    def add_action(self, action):
        assert not self.completed
        self.actions.append(action)
    
    def add_reward(self, reward):
        assert not self.completed
        self.rewards.append(reward)
        
    def add_step_info(self, step_info):
        assert not self.completed
        self.step_infos.append(step_info)
        if step_info.last:
            assert self.completed
    
    @property
    def last_observation(self):
        return self.observations[-1]
    
    @property
    def numpy_observations(self):
        return np.concatenate(self.observations, axis=0)
    
    @property
    def numpy_actions(self):
        return np.concatenate(self.actions, axis=0)
        
    @property
    def numpy_rewards(self):
        return np.asarray(self.rewards)
    
    @property
    def numpy_dones(self):
        return np.asarray([step_info.done for step_info in self.step_infos])
    
    @property
    def numpy_masks(self):
        return 1. - self.numpy_dones
    
    @property
    def infos(self):
        return [step_info.info for step_info in self.step_infos]
    
    def get_all_info(self, key):
        return [step_info[key] for step_info in self.step_infos]
    
    def __repr__(self):
        return f'Trajectory(T: {len(self)}, Completed: {self.completed}, Reach time limit: {self.reach_time_limit}, Reach terminal: {self.reach_terminal})'
