import numpy as np

from collections import OrderedDict

from lagom.envs.spaces import Box, Dict
from .base import VecEnv


class DummyVecEnv(VecEnv):
    def __init__(self, list_make_env):
        self.envs = [make_env() for make_env in list_make_env]
        
        observation_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space
        super().__init__(len(list_make_env), observation_space, action_space)
        
        assert isinstance(self.observation_space, (Box, Dict))  # enforce observation space either Box or Dict
        
    def step_async(self, actions):
        self.actions = actions
        
    def step_wait(self):
        outputs = []
        for env, action in zip(self.envs, self.actions):
            observation, reward, done, info = env.step(action)
            if done:
                observation = env.reset()
            outputs.append([observation, reward, done, info])
        observations, rewards, dones, infos = zip(*outputs)
        
        observations = self._process_observations(observations)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        return observations, rewards, dones, infos
        
    def reset(self):
        observations = [env.reset() for env in self.envs]
        return self._process_observations(observations)
    
    def close(self):
        return
    
    def _process_observations(self, observations):
        if isinstance(self.observation_space, Box):
            return np.stack(observations)
        elif isinstance(self.observation_space, Dict):
            spaces = self.observation_space.spaces
            outputs = []
            for key in spaces.keys():
                outputs.append((key, np.stack([observation[key] for observation in observations], axis=0)))
            return OrderedDict(outputs)
        else:
            raise TypeError('Only Box and Dict are supported. ')
