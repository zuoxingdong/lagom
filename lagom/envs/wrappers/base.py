from lagom.envs import Env


class Wrapper(Env):
    def __init__(self, env):
        if not isinstance(env, Env):
            raise TypeError('The object env must be of type lagom.envs.Env.')
            
        self.env = env
        
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        return self.env.reset()
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def clean(self):
        return self.env.clean()
    
    def get_source_env(self):
        return self.env.get_source_env()
    
    @property
    def T(self):
        return self.env.T
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    

class ObservationWrapper(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        return self.process_observation(observation), reward, done, info
    
    def reset(self):
        observation = self.env.reset()
        
        return self.process_observation(observation)
    
    def process_observation(self, observation):
        raise NotImplementedError
        
    @property
    def observation_space(self):
        raise NotImplementedError
        
        
class ActionWrapper(Wrapper):
    def step(self, action):
        return self.env.step(self.process_action(action))
    
    def process_action(self, action):
        raise NotImplementedError
        
    @property
    def action_space(self):
        raise NotImplementedError
        
        
class RewardWrapper(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        return observation, self.process_reward(reward), done, info
    
    def process_reward(self, reward):
        raise NotImplementedError