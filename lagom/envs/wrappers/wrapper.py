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
    
    def close(self):
        return self.env.close()
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)
    
    @property
    def unwrapped(self):
        return self.env.unwrapped
    
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
    """
    Observation wrapper modifies the received observation after each function call of step() and reset(). 
    
    Note that when observation is modified, the observation space should also be adjusted accordingly.
    
    All inherited subclasses should at least implement the following functions:
    1. process_observation(self, observation)
    2. @property: observation_space(self)
    """
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
    """
    Action wrapper modifies the action before calling step() function. 
    
    Note that when action is modified, the action space should also be adjusted accordingly. 
    
    All inherited subclasses should at least implement the following functions:
    1. process_action(self, action)
    2. @property: action_space(self)
    """
    def step(self, action):
        return self.env.step(self.process_action(action))
    
    def process_action(self, action):
        raise NotImplementedError
        
    @property
    def action_space(self):
        raise NotImplementedError
        
        
class RewardWrapper(Wrapper):
    """
    Reward wrapper modifies the received reward after each function call of step(). 
    
    All inherited subclasses should at least implement the following functions:
    1. process_reward(self, reward)
    """
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        return observation, self.process_reward(reward), done, info
    
    def process_reward(self, reward):
        raise NotImplementedError
