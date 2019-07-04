from functools import partial  # argument-free functions

from lagom.utils import Seeder
from lagom.utils import CloudpickleWrapper

from .vec_env import VecEnv


def make_vec_env(make_env, num_env, init_seed):
    r"""Create a vectorized environment, each associated with a different random seed.
    
    Example::
        
        >>> import gym
        >>> make_vec_env(lambda: gym.make('CartPole-v1'), 3, 0)
        <VecEnv: 3, CartPole-v1>
    
    Args:
        make_env (function): a function to create an environment
        num_env (int): number of environments to create. 
        init_seed (int): initial seed for :class:`Seeder` to sample random seeds. 
    
    Returns
    -------
    env : VecEnv
        created vectorized environment
    
    """
    # Generate different seeds for each environment
    seeder = Seeder(init_seed=init_seed)
    seeds = seeder(size=num_env)
    
    def f(seed):
        env = make_env()
        env.seed(seed)
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        return env
    
    # Use partial to generate a list of argument-free make_env, each with different seed
    # partial object is not picklable, so wrap it with magical CloudpickleWrapper
    list_make_env = [CloudpickleWrapper(partial(f, seed=seed)) for seed in seeds]
    return VecEnv(list_make_env)
