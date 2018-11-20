import numpy as np

from .vec_env import VecEnvWrapper


class VecClipAction(VecEnvWrapper):
    r"""A vectorized environment wrapper that clips the given action within a valid
    bound before feeding the environment step function. 
    
    .. note::
        Action clipping is valid only for continuous action space. 
    """
    def step_async(self, actions):
        low = self.action_space.low
        high = self.action_space.high
        actions = [np.clip(action, low, high) for action in actions]
        
        self.venv.step_async(actions)
    
    def step_wait(self):
        return self.venv.step_wait()
    
    def reset(self):
        return self.venv.reset()
    
    def close_extras(self):
        return self.venv.close_extras()
