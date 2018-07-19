import numpy as np

from lagom.envs.vec_env import VecEnv


class LinearVecEnv(VecEnv):
    """
    Run vectorized environment linearly. 
    
    Note that it is recommended to use linear vectorized environment only if step() in the environment
    needs very few computation, otherwise it will be slower than doing it parallelly. For 'slow' environment, 
    it is recommended to use ParallelVecEnv instead. 
    
    Examples:
        def make_env():
            env = gym.make('CartPole-v0')
            env = GymEnv(env)

            return env

        env = LinearVecEnv([make_env]*5)
        env.reset()
        for _ in range(100):
            env.step([0]*5)
    """
    def __init__(self, list_make_env):
        """
        Args:
            list_make_env (list): list of functions to generate an environment. 
        """
        # Create list of environments
        self.list_env = [make_env() for make_env in list_make_env]
        
        # Call parent constructor
        super().__init__(list_make_env=list_make_env, 
                         observation_space=self.list_env[0].observation_space, 
                         action_space=self.list_env[0].action_space)
        
    def step_async(self, actions):
        # Record current actions
        self.actions = actions
        
    def step_wait(self):
        # Execute the recorded actions, each for one environment
        observations = []
        rewards = []
        dones = []
        infos = []
        for env, action in zip(self.list_env, self.actions):
            observation, reward, done, info = env.step(action)
            
            # If episode terminates, reset the environment and record initial observation in info
            if done:
                init_observation = env.reset()
                info['init_observation'] = init_observation
                
            # Record all information
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return observations, rewards, dones, infos
    
    def reset(self):
        # Reset all the environment and return all the initial observations
        observations = [env.reset() for env in self.list_env]
        
        return observations
    
    def render(self, mode='human'):
        # Render all the environments and return rendered images
        imgs = [env.render(mode='rgb_array') for env in self.list_env]
        
        return imgs
    
    def close(self):
        # Close all the environments
        [env.close() for env in self.list_env]
        
    def seed(self, seeds):
        # Seed all the environments with given seeds
        [env.seed(seed) for env, seed in zip(self.list_env, seeds)]
        
    @property
    def T(self):
        all_T = [env.T for env in self.list_env]
        
        return all_T