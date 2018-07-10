from .base_policy import BasePolicy

class RandomPolicy(BasePolicy):
    """
    A random policy. The action is sampled from action space.
    """
    def __call__(self, x):
        # Randomly sample an action from action space
        action = self.env_spec.action_space.sample()
        
        # Dictionary of output
        out = {}
        out['action'] = action
        
        return out