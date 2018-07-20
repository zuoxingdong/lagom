from abc import abstractmethod

from .vec_env import VecEnv


class VecEnvWrapper(VecEnv):
    def __init__(self, venv):
        self.venv = venv
        
        super().__init__(list_make_env=venv.list_make_env, 
                         observation_space=venv.observation_space, 
                         action_space=venv.action_space)
        
    def step_async(self, actions):
        self.venv.step_async(actions)
        
    @abstractmethod
    def step_wait(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    def render(self, mode='human'):
        return self.venv.render(mode)
    
    def close(self):
        return self.venv.close()
    
    def seed(self, seeds):
        return self.venv.seed(seeds)
    
    @property
    def unwrapped(self):
        return self.venv.unwrapped
    
    @property
    def T(self):
        return self.venv.T

    @property
    def observation_space(self):
        """
        Return a Space object to define the observation space.
        """
        return self.venv.observation_space
    
    @property
    def action_space(self):
        """
        Return a Space object to define the action space.
        """
        return self.venv.action_space
