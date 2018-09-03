from .base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    r"""Random policy. 
    
    The action is uniformly sampled from action space.
    
    Example::
    
        from lagom.envs import make_gym_env
        from lagom.envs import make_envs
        from lagom.envs import EnvSpec
        from lagom.envs.vec_env import SerialVecEnv

        from lagom.core.policies import RandomPolicy

        list_make_env = make_envs(make_env=make_gym_env, 
                                  env_id='CartPole-v1', 
                                  num_env=2, 
                                  init_seed=0)
        venv = SerialVecEnv(list_make_env=list_make_env)
        obs = venv.reset()

        env_spec = EnvSpec(venv)
        policy = RandomPolicy(config=None, network=None, env_spec=env_spec)
        policy(obs)
        
    """
    def __call__(self, x):
        out_policy = {}
        
        # Randomly sample an batched action from action space for VecEnv
        action = [self.env_spec.action_space.sample() for _ in range(self.env_spec.env.num_env)]
        
        # Record output
        out_policy['action'] = action
        
        return out_policy
