from dataclasses import dataclass
from lagom.envs import VecEnvWrapper


@dataclass
class StepInfo:
    r"""Defines a set of information for each time step. 
    
    A `StepInfo` is returned from each `step` and `reset` of an environment. 
    It contains properties of the transition and additional information. 
    
    """
    done: bool
    info: dict
    
    def __getitem__(self, key):
        return self.info[key]
    
    @property
    def first(self):
        return self.info.get('FIRST', False)
    
    @property
    def mid(self):
        return not self.first and not self.done
        
    @property
    def last(self):
        return self.done
        
    @property
    def time_limit(self):
        return self.info.get('TimeLimit.truncated', False)
        
    @property
    def terminal(self):
        return self.done and not self.time_limit


class VecStepInfo(VecEnvWrapper):
    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)
        step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
        return observations, rewards, step_infos
        
    def reset(self):
        observations = self.env.reset()
        step_infos = [StepInfo(False, {'FIRST': True}) for _ in range(len(observations))]
        return observations, step_infos
