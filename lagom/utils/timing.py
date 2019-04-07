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
    timestamp = datetime.now().isoformat(' ', 'seconds')
    print(color_str(string=f'\nTotal time: {timedelta(seconds=round(perf_counter() - t))} at {timestamp}', 
                    color=color, 
                    attribute=attribute))
