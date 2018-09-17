from .base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    r"""Random policy. 
    
    The action is uniformly sampled from action space.
    
    Example::
    
        policy = RandomPolicy(config=None, network=None, env_spec=env_spec)
        policy(observation)
        
    """
    def __call__(self, x, out_keys=['action']):
        out_policy = {}
        
        # Randomly sample an batched action from action space for VecEnv
        action = [self.env_spec.action_space.sample() for _ in range(self.env_spec.env.num_env)]
        
        # Record output
        out_policy['action'] = action
        
        return out_policy
