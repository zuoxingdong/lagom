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
    
    Example::
    
        >>> from lagom.envs import make_envs, make_gym_env
        >>> list_make_env = make_envs(make_env=make_gym_env, env_id='CartPole-v1', num_env=3, init_seed=0)
        >>> env = SerialVecEnv(list_make_env=list_make_env)
        >>> env
        <SerialVecEnv: CartPole-v1, n: 3>
        
        >>> env.reset()
        [array([-0.04002427,  0.00464987, -0.01704236, -0.03673052]),
         array([ 0.00854682,  0.00830137, -0.03052506,  0.03439879]),
         array([0.00025361, 0.02915667, 0.01103413, 0.04977449])]
    
    """
    def __init__(self, list_make_env):
        self.list_env = [make_env() for make_env in list_make_env]
        super().__init__(list_make_env=list_make_env, 
                         observation_space=self.list_env[0].observation_space, 
                         action_space=self.list_env[0].action_space, 
                         reward_range=self.list_env[0].reward_range, 
                         spec=self.list_env[0].spec)
        
    def step_async(self, actions):
        assert len(actions) == self.num_env, f'expected length {self.num_env}, got {len(actions)}'
        self.actions = actions  # Record as current actions
        
    def step_wait(self):
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.list_env, self.actions)):
            observation, reward, done, info = env.step(action)
            # If episode terminates, reset this environment and report initial observation for new episode
            # the terminal observation is stored in the info
            if done:
                info['terminal_observation'] = observation
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
