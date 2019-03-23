def get_wrapper(env, name):
    r"""Return a wrapped environment of a specific wrapper. 
    
    .. note::
        If no such wrapper found, then an ``None`` is returned. 
    
    Args:
        env (Env): environment
        name (str): name of the wrapper
        
    Returns
    -------
    out : env
        wrapped environment
    """
    if name == env.__class__.__name__:
        return env
    elif env.unwrapped is env:  # reaching underlying environment
        return None
    else:
        return get_wrapper(env.unwrapped, name)
