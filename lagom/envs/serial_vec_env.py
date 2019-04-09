import numpy as np

from .vec_env import VecEnv


class SerialVecEnv(VecEnv):
    r"""A vectorized environment runs serially for each sub-environment. 
    
    For each :meth:`step` and :meth:`reset`, the command is executed in one sub-environment
    at a time. 
    
    .. note::
    
        It is recommended to use this if the simulator is very fast. In this case, :class:`ParallelVecEnv`
        would have too much computation overheads which might even slow down the speed.
        However, if the simulator is very computationally expensive, one should use
        :class:`ParallelVecEnv` instead. 
    
    """
    def __init__(self, list_make_env):
        self.list_env = [make_env() for make_env in list_make_env]
        super().__init__(list_make_env=list_make_env, 
                         observation_space=self.list_env[0].observation_space, 
                         action_space=self.list_env[0].action_space, 
                         reward_range=self.list_env[0].reward_range, 
                         spec=self.list_env[0].spec)
    
    def step(self, actions):
        assert len(actions) == len(self)
        observations = []
        rewards = []
        dones = []
        infos = []
        for i, (env, action) in enumerate(zip(self.list_env, actions)):
            observation, reward, done, info = env.step(action)
            # If done=True, reset environment, store last observation in info and report new initial observation
            if done:
                info['last_observation'] = observation
                observation = env.reset()
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return observations, rewards, dones, infos
    
    def reset(self):
        observations = [env.reset() for env in self.list_env]
        return observations
    
    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.list_env]
    
    def close_extras(self):
        return [env.close() for env in self.list_env]
    
    def __getitem__(self, index):
        return self.list_env[index]
    
    def __setitem__(self, index, x):
        self.list_env[index] = x
