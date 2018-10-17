from contextlib import contextmanager

from time import time
from datetime import timedelta

from .colorize import color_str


@contextmanager
def timed(color='green', attribute='bold'):
    r"""A decorator to print the total time of executing a body function. 
    
    Args:
        color (str, optional): color name. Default: 'green'
        attribute (str, optional): attribute. Default: 'bold'
    """
    t = time()
    yield
    print(color_str(string=f'\nTotal time: {timedelta(seconds=round(time() - t))}', 
                    color=color, 
                    attribute=attribute))
