from .base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    r"""Random policy. 
    
    The action is uniformly sampled from action space.
    
    Example::
    
        policy = RandomPolicy(config=None, env_spec=env_spec, device=None)
        policy(observation)
        
    """
    def __init__(self, config, env_spec):
        self.config = config
        self._env_spec = env_spec
        
    def make_networks(self, config):
        pass
    
    def reset(self, config, **kwargs):
        pass
    
    def __call__(self, x, out_keys=['action'], info={}, **kwargs):
        out_policy = {}
        
        if self.env_spec.is_vec_env:
            action = [self.action_space.sample() for _ in range(self.env_spec.env.num_env)]
        else:
            action = self.action_space.sample()
        
        out_policy['action'] = action
        
        return out_policy
    
    @property
    def recurrent(self):
        pass
