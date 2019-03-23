from functools import partial  # argument-free functions

from lagom.utils import Seeder

from .serial_vec_env import SerialVecEnv
from .parallel_vec_env import ParallelVecEnv


def make_vec_env(make_env, num_env, init_seed, mode='serial'):
    r"""Create a vectorized environment, each associated with a different random seed.
    
    Example::
        
        >>> import gym
        >>> make_vec_env(lambda: gym.make('CartPole-v1'), 3, 0)
        <SerialVecEnv: 3, CartPole-v1>
    
    Args:
        make_env (function): a function to create an environment
        num_env (int): number of environments to create. 
        init_seed (int): initial seed for :class:`Seeder` to sample random seeds. 
        mode (str, optional): specifies the type of vectorized environment ['serial', 'parallel'].
            'serial': uses :class:`SerialVecEnv`. 'parallel': uses :class:`ParallelVecEnv`
    
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
        return env
    
    # Use partial to generate a list of argument-free make_env, each with different seed
    list_make_env = [partial(f, seed=seed) for seed in seeds]
    
    assert mode in ['serial', 'parallel']
    if mode == 'serial':
        return SerialVecEnv(list_make_env)
    elif mode == 'parallel':
        return ParallelVecEnv(list_make_env)
