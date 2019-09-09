from enum import IntEnum
from dataclasses import dataclass
import numpy as np


class StepType(IntEnum):
    FIRST = 0
    MID = 1
    LAST = 2


@dataclass
class TimeStep:
    step_type: StepType
    observation: object
    reward: float
    done: bool
    info: dict
    
    def __getitem__(self, key):
        return self.info[key]
    
    def first(self):
        if self.step_type == StepType.FIRST:
            assert all([x is None for x in [self.reward, self.done, self.info]])
        return self.step_type == StepType.FIRST
    
    def mid(self):
        if self.step_type == StepType.MID:
            assert not self.first() and not self.last()
        return self.step_type == StepType.MID
        
    def last(self):
        if self.step_type == StepType.LAST:
            assert self.done is not None and self.done
        return self.step_type == StepType.LAST
        
    def time_limit(self):
        return self.last() and self.info.get('TimeLimit.truncated', False)
    
    def terminal(self):
        return self.last() and not self.time_limit()
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.step_type.name})'


class Trajectory(object):
    def __init__(self):
        self.timesteps = []
        self._actions = []
        self._extra_info = {}
        
    def __len__(self):
        return len(self.timesteps)
    
    @property
    def T(self):
        return max(0, len(self) - 1)
    
    def __getitem__(self, index):
        return self.timesteps[index]
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i < len(self):
            timestep = self.timesteps[self.i]
            self.i += 1
            return timestep
        else:
            raise StopIteration
    
    @property
    def finished(self):
        return len(self) > 0 and self.timesteps[-1].last()
    
    @property
    def reach_time_limit(self):
        return len(self) > 0 and self.timesteps[-1].time_limit()
    
    @property
    def reach_terminal(self):
        return len(self) > 0 and self.timesteps[-1].terminal()

    def add(self, timestep, action):
        assert not self.finished
        if len(self) == 0:
            assert timestep.first()
            assert action is None
        else:
            assert action is not None
            self._actions.append(action)
        self.timesteps.append(timestep)

    @property
    def observations(self):
        return [timestep.observation for timestep in self.timesteps]
    
    @property
    def actions(self):
        return self._actions
        
    @property
    def rewards(self):
        return [timestep.reward for timestep in self.timesteps[1:]]
    
    @property
    def dones(self):
        return [timestep.done for timestep in self.timesteps[1:]]
    
    @property
    def infos(self):
        return [timestep.info for timestep in self.timesteps[1:]]
    
    def get_infos(self, key):
        return [timestep.info[key] for timestep in self.timesteps[1:] if key in timestep.info]
    
    @property
    def extra_info(self):
        return self._extra_info
    
    @extra_info.setter
    def extra_info(self, info):
        self._extra_info = info
    
    def __repr__(self):
        return f'Trajectory(T: {self.T}, Finished: {self.finished}, Reach time limit: {self.reach_time_limit}, Reach terminal: {self.reach_terminal})'
