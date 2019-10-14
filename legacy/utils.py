def get_wrapper(env, name):
    r"""Return a wrapped environment of a specific wrapper. 
    
    .. note::
        If no such wrapper found, then an ``None`` is returned. 
    
    Args:
        env (Env): environment
        name (str): name of the wrapper
        
    Returns:
        Env: wrapped environment
    """
    if name == env.__class__.__name__:
        return env
    elif env.unwrapped is env:  # reaching underlying environment
        return None
    else:
        return get_wrapper(env.env, name)


def get_all_wrappers(env):
    r"""Returns a list of wrapper names of a wrapped environment. 
    
    Args:
        env (Env): wrapped environment
    
    Returns:
        list: a list of string names of wrappers
    """
    out = []
    while env is not env.unwrapped:
        out.append(env.__class__.__name__)
        env = env.env
    return out
