from gym import Wrapper

from lagom.envs import VecEnvWrapper


def get_wrapper(env, name):
    r"""Return a wrapper of an environment by its name. 
    
    .. note::
        If no such wrapper found, then an ``None`` is returned. 
    
    Args:
        env (VecEnv): vectorized environment. 
        name (str): name of the wrapper
        
    Returns
    -------
    out : VecEnvWrapper
        wrapper of the environment
    """
    while True:
        if name == env.__class__.__name__:
            return env
        elif isinstance(env, Wrapper):
            env = env.env
        elif isinstance(env, VecEnvWrapper):
            env = env.venv
        else:
            return None
