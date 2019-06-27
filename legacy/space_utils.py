import numpy as np

from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import MultiDiscrete
from gym.spaces import MultiBinary
from gym.spaces import Tuple
from gym.spaces import Dict


def flatdim(space):
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return int(space.n)
    elif isinstance(space, Tuple):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([flatdim(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return int(np.prod(space.shape))
    else:
        raise NotImplementedError


def flatten(space, x):
    if isinstance(space, Box):
        return np.asarray(x, dtype=np.float32).flatten()
    elif isinstance(space, Discrete):
        onehot = np.zeros(space.n, dtype=np.float32)
        onehot[x] = 1.0
        return onehot
    elif isinstance(space, Tuple):
        return np.concatenate([flatten(s, x_part) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, Dict):
        return np.concatenate([flatten(space.spaces[key], item) for key, item in x.items()])
    elif isinstance(space, MultiBinary):
        return np.asarray(x).flatten()
    elif isinstance(space, MultiDiscrete):
        return np.asarray(x).flatten()
    else:
        raise NotImplementedError


def unflatten(space, x):
    if isinstance(space, Box):
        return np.asarray(x, dtype=np.float32).reshape(space.shape)
    elif isinstance(space, Discrete):
        return int(np.nonzero(x)[0][0])
    elif isinstance(space, Tuple):
        dims = [flatdim(s) for s in space.spaces]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [unflatten(s, flattened) 
                            for flattened, s in zip(list_flattened, space.spaces)]
        return tuple(list_unflattened)
    elif isinstance(space, Dict):
        dims = [flatdim(s) for s in space.spaces.values()]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [(key, unflatten(s, flattened)) 
                            for flattened, (key, s) in zip(list_flattened, space.spaces.items())]
        return dict(list_unflattened)
    elif isinstance(space, MultiBinary):
        return np.asarray(x).reshape(space.shape)
    elif isinstance(space, MultiDiscrete):
        return np.asarray(x).reshape(space.shape)
    else:
        raise NotImplementedError


def test_space_utils():
    # Box
    box = Box(-1.0, 1.0, shape=[2, 3], dtype=np.float32)
    sample = box.sample()
    assert flatdim(box) == 2*3
    assert flatten(box, sample).shape == (2*3,)
    assert np.allclose(sample, unflatten(box, flatten(box, sample)))

    x = np.array([[1.0, 1.0], [1.0, 1.0]])
    box = Box(low=-x, high=x, dtype=np.float32)
    sample = box.sample()
    assert flatdim(box) == 2*2
    assert flatten(box, sample).shape == (2*2,)
    assert np.allclose(sample, unflatten(box, flatten(box, sample)))

    # Discrete
    discrete = Discrete(5)
    sample = discrete.sample()
    assert flatdim(discrete) == 5
    assert flatten(discrete, sample).shape == (5,)
    assert sample == unflatten(discrete, flatten(discrete, sample))

    # Tuple
    S = Tuple([Discrete(5), 
               Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32), 
               Dict({'success': Discrete(2), 'velocity': Box(-1, 1, shape=(1, 3), dtype=np.float32)})])
    sample = S.sample()
    assert flatdim(S) == 5+2*3+2+3
    assert flatten(S, sample).shape == (16,)
    _sample = unflatten(S, flatten(S, sample))
    assert sample[0] == _sample[0]
    assert np.allclose(sample[1], _sample[1])
    assert sample[2]['success'] == _sample[2]['success']
    assert np.allclose(sample[2]['velocity'], _sample[2]['velocity'])

    # Dict
    D0 = Dict({'position': Box(-100, 100, shape=(3,), dtype=np.float32), 
               'velocity': Box(-1, 1, shape=(4,), dtype=np.float32)})
    D = Dict({'sensors': D0, 'score': Discrete(100)})
    sample = D.sample()
    assert flatdim(D) == 3+4+100
    assert flatten(D, sample).shape == (107,)
    _sample = unflatten(D, flatten(D, sample))
    assert sample['score'] == _sample['score']
    assert np.allclose(sample['sensors']['position'], _sample['sensors']['position'])
    assert np.allclose(sample['sensors']['velocity'], _sample['sensors']['velocity'])
