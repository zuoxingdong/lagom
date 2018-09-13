import gym

from .discrete import Discrete
from .box import Box
from .product import Product
from .dict import Dict


def convert_gym_space(space):
    r"""Convert an OpenAI Gym Space object to lagom :class:`Space`. 

    Args:
        space (Gym Space): a space object from OpenAI Gym environment. 

    Returns
    -------
    out : Space
        converted lagom-compatible space
    """
    if isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high, dtype=space.dtype)  # don't give shape
    elif isinstance(space, gym.spaces.Tuple):
        return Product([convert_gym_space(s) for s in space.spaces])
    elif isinstance(space, gym.spaces.Dict):
        return Dict({k: convert_gym_space(s) for k, s in space.spaces.items()})
    else:
        raise TypeError(f'expected type as Discrete, Box, Tuple or Dict, got {type(space)}')
