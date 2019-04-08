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
        
    def __repr__(self):
        ind = '\t'*self.repr_indent
        s = ind + f'count: {self.count}\n'
        s += ind + f'mean: {self.mean}\n'
        s += ind + f'std: {self.std}\n'
        s += ind + f'min: {self.min}\n'
        s += ind + f'max: {self.max}'
        return s


def describe(x, axis=-1, repr_indent=0):
    count = np.shape(x)[-1]
    mean = np.mean(x, axis)
    std = np.std(x, axis)
    min = np.min(x, axis)
    max = np.max(x, axis)
    return Describe(count, mean, std, min, max, repr_indent)
