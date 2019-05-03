import os

import numpy as np
import torch

from lagom.utils import color_str
from lagom.utils import Seeder
from lagom.utils import tensorify
from lagom.utils import numpify
from lagom.utils import pickle_load
from lagom.utils import pickle_dump
from lagom.utils import yaml_load
from lagom.utils import yaml_dump
from lagom.utils import timed


def test_color_str():
    assert color_str('lagom', 'green', 'bold') == '\x1b[38;5;2m\x1b[1mlagom\x1b[0m'
    assert color_str('lagom', 'white') == '\x1b[38;5;15mlagom\x1b[0m'

    
def test_seeder():
    seeder = Seeder(init_seed=0)

    assert seeder.rng.get_state()[1][0] == 0
    assert np.random.get_state()[1][20] != seeder.rng.get_state()[1][20]

    # Single list of seeds
    seeds = seeder(size=1)
    assert len(seeds) == 1
    seeds = seeder(size=5)
    assert len(seeds) == 5

    # Batched seeds
    seeds = seeder(size=[1, 3])
    assert np.alltrue(np.array(seeds).shape == (1, 3))
    seeds = seeder(size=[2, 3])
    assert np.alltrue(np.array(seeds).shape == (2, 3))

    
def test_tensorify():
    # tensor
    x = torch.tensor(2.43)
    y = tensorify(x, 'cpu')
    assert torch.equal(x, y)
    del x, y

    x = torch.randn(10)
    y = tensorify(x, 'cpu')
    assert torch.equal(x, y)
    del x, y

    x = torch.randn(10, 20, 30)
    y = tensorify(x, 'cpu')
    assert torch.equal(x, y)
    del x, y

    # ndarray
    x = np.array(2.43)
    y = tensorify(x, 'cpu')
    assert np.allclose(x, y.item())
    del x, y

    x = np.random.randn(10)
    y = tensorify(x, 'cpu')
    assert np.allclose(x, y)
    del x, y

    x = np.random.randn(10, 20, 30)
    y = tensorify(x, 'cpu')
    assert np.allclose(x, y)
    del x, y

    # raw list
    x = [2.43]
    y = tensorify(x, 'cpu')
    assert np.allclose(x, y.item())
    del x, y

    x = [1, 2, 3, 4, 5, 6]
    y = tensorify(x, 'cpu')
    assert np.allclose(x, y)
    del x, y

    x = [[1, 2], [3, 4], [5, 6]]
    y = tensorify(x, 'cpu')
    assert np.allclose(x, y)
    del x, y


def test_numpify():
    # tensor
    x = torch.tensor(2.43)
    y = numpify(x, np.float32)
    assert np.allclose(x, y)
    del x, y

    x = torch.randn(10)
    y = numpify(x, np.float32)
    assert np.allclose(x, y)
    del x, y

    x = torch.randn(10, 20, 30)
    y = numpify(x, np.float32)
    assert np.allclose(x, y)
    del x, y

    # ndarray
    x = np.array(2.43)
    y = numpify(x, np.float32)
    assert np.allclose(x, y)
    del x, y

    x = np.random.randn(10)
    y = numpify(x, np.float32)
    assert np.allclose(x, y)
    del x, y

    x = np.random.randn(10, 20, 30)
    y = numpify(x, np.float32)
    assert np.allclose(x, y)
    del x, y

    # raw list
    x = [2.43]
    y = numpify(x, np.float32)
    assert np.allclose(x, y)
    del x, y

    x = [1, 2, 3, 4, 5, 6]
    y = numpify(x, np.float32)
    assert np.allclose(x, y)
    del x, y

    x = [[1, 2], [3, 4], [5, 6]]
    y = numpify(x, np.float32)
    assert np.allclose(x, y)
    del x, y
    

def test_pickle_yaml():
    a = {'one': 1, 'two': [2, 3]}
    b = {'three': 3, 'four': [4, 5]}
    c = [a, b]

    def check(x):
        assert isinstance(x, list)
        assert len(x) == 2
        assert all([isinstance(i, dict) for i in x])
        assert list(x[0].keys()) == ['one', 'two']
        assert list(x[1].keys()) == ['three', 'four']
        assert list(x[0].values()) == [1, [2, 3]]
        assert list(x[1].values()) == [3, [4, 5]]

    pickle_dump(c, '.tmp_pickle')
    check(pickle_load('.tmp_pickle.pkl'))
    os.unlink('.tmp_pickle.pkl')

    yaml_dump(c, '.tmp_yaml')
    check(yaml_load('.tmp_yaml.yml'))
    os.unlink('.tmp_yaml.yml')


def test_timed():
    with timed('red', 'bold'):
        a = 5
