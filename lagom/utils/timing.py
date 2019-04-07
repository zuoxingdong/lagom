import functools
from contextlib import contextmanager

from time import perf_counter
from datetime import timedelta
from datetime import datetime

from .colorize import color_str


@contextmanager
def timed(color='green', attribute='bold'):
    r"""A decorator to print the total time of executing a body function. 
    
    Args:
        color (str, optional): color name. Default: 'green'
        attribute (str, optional): attribute. Default: 'bold'
    """
    t = perf_counter()
    yield
    total_time = timedelta(seconds=round(perf_counter() - t))
    timestamp = datetime.now().isoformat(' ', 'seconds')
    print(color_str(string=f'\nTotal time: {total_time} at {timestamp}', 
                    color=color, 
                    attribute=attribute))
    
    
def timeit(_func=None, *, color='green', attribute='bold'):
    def decorator_timeit(f):
        r"""Print the runtime of the decorated function. """
        @functools.wraps(f)
        def wrapper_timeit(*args, **kwargs):
            t = perf_counter()
            out = f(*args, **kwargs)
            total_time = timedelta(seconds=round(perf_counter() - t))
            timestamp = datetime.now().isoformat(' ', 'seconds')
            print(color_str(string=f'\nTotal time: {total_time} at {timestamp}', 
                            color=color, 
                            attribute=attribute))
            return out
        return wrapper_timeit
    if _func is None:
        return decorator_timeit
    else:
        return decorator_timeit(_func)
