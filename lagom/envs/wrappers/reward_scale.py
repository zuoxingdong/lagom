from lagom.envs.wrappers import RewardWrapper


class RewardScale(RewardWrapper):
    r"""Scale the reward. 
    
    .. note::
    
        This is incredibly important and drastically impact on performance e.g. PPO. 
        
    Example::
    
        >>> env = make_gym_env(env_id='CartPole-v1', seed=0)
        >>> env = RewardScale(env, scale=0.1)
        >>> env.reset()
        >>> observation, reward, done, info = env.step(env.action_space.sample())
        >>> reward
        0.1
        
    """
    def __init__(self, env, scale=0.01):
        r"""Initialize the wrapper. 
        
        Args:
            env (Env): environment
            scale (float): reward scaling factor
        """
        super().__init__(env=env)
        self.scale = scale
        
    def process_reward(self, reward):
        return self.scale*reward
