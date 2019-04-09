from dataclasses import dataclass
import numpy as np


@dataclass
class Describe:
    count: int
    mean: float
    std: float
    min: float
    max: float
    repr_indent: int = 0
    repr_prefix: str = None
        
    def __repr__(self):
        s = ''
        if self.repr_prefix is not None:
            s += self.repr_prefix
        ind = '\t'*self.repr_indent
        s += ind + f'count: {self.count}\n'
        s += ind + f'mean: {self.mean}\n'
        s += ind + f'std: {self.std}\n'
        s += ind + f'min: {self.min}\n'
        s += ind + f'max: {self.max}'
        return s


def describe(x, axis=-1, repr_indent=0, repr_prefix=None):
    if x is None or np.size(x) == 0:
        return None
    x = np.asarray(x)
    count = x.shape[-1]
    mean = x.mean(axis)
    std = x.std(axis)
    min = x.min(axis)
    max = x.max(axis)
    return Describe(count, mean, std, min, max, repr_indent, repr_prefix)
