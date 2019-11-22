import os
import pytest

import numpy as np
import torch

from lagom.utils import color_str
from lagom.utils import IntervalConditioner
from lagom.utils import NConditioner
from lagom.utils import Seeder
from lagom.utils import tensorify
from lagom.utils import numpify
from lagom.utils import pickle_load
from lagom.utils import pickle_dump
from lagom.utils import yaml_load
from lagom.utils import yaml_dump
from lagom.utils import timed
from lagom.utils import ProcessMaster
from lagom.utils import ProcessWorker


def test_color_str():
    assert color_str('lagom', 'green', bold=True) == '\x1b[32m\x1b[1mlagom\x1b[0m'
    assert color_str('lagom', 'white') == '\x1b[37mlagom\x1b[0m'


def test_conditioner():
    cond = IntervalConditioner(interval=4, mode='accumulative')
    assert cond.counter == 0
    assert cond(0)
    assert cond.counter == 0
    assert not cond(2)
    assert not cond(3)
    assert cond(4)
    assert cond.counter == 1
    assert not cond(5)
    assert cond(9)
    assert cond.counter == 2
    assert cond(12)
    assert cond.counter == 3
    del cond

    cond = IntervalConditioner(interval=4, mode='incremental')
    assert cond.counter == 0
    assert cond.total_n == 0
    assert cond(0)
    assert cond.counter == 0
    assert cond.total_n == 0
    assert not cond(3)
    assert cond.counter == 0
    assert cond.total_n == 3
    assert cond(1)
    assert cond.counter == 1
    assert cond.total_n == 4
    assert cond(4)
    assert cond.counter == 2
    assert cond.total_n == 8
    assert not cond(1)
    assert cond.counter == 2
    assert cond.total_n == 9
    del cond

    cond = NConditioner(max_n=10, num_conditions=3, mode='accumulative')
    assert cond.counter == 0
    assert cond(0)
    assert cond.counter == 0
    assert not cond(2)
    assert not cond(3)
    assert cond(4)
    assert cond.counter == 1
    assert not cond(5)
    assert cond.counter == 1
    assert cond(8)
    assert cond.counter == 2
    assert not cond(9)
    assert cond.counter == 2
    assert cond(10)
    assert cond.counter == 3
    assert not cond(15)
    assert cond.counter == 3
    assert not cond(20)
    assert cond.counter == 3
    del cond

    cond = NConditioner(max_n=10, num_conditions=3, mode='incremental')
    assert cond.counter == 0
    assert cond(0)
    assert cond.counter == 0
    assert not cond(2)
    assert cond.counter == 0
    assert cond.total_n == 2
    assert not cond(1)
    assert cond.counter == 0
    assert cond.total_n == 3
    assert cond(2)
    assert cond.counter == 1
    assert cond.total_n == 5
    assert not cond(1)
    assert cond.counter == 1
    assert cond.total_n == 6
    assert cond(2)
    assert cond.counter == 2
    assert cond.total_n == 8
    assert cond(3)
    assert cond.counter == 3
    assert cond.total_n == 11
    assert not cond(5)
    assert cond.counter == 3
    assert cond.total_n == 11
    del cond


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
    # Tensor
    x = torch.randn(5)
    y = numpify(x)
    assert isinstance(y, np.ndarray)
    assert x.shape == y.shape
    del x, y

    x = torch.randn(5, 4)
    y = numpify(x)
    assert isinstance(y, np.ndarray)
    assert x.shape == y.shape
    del x, y

    x = torch.randn(5, 4)
    y = numpify(x, dtype=np.float16)
    assert isinstance(y, np.ndarray)
    assert x.shape == y.shape
    assert y.dtype == np.float16
    del x, y

    # Array
    x = np.random.randn(5)
    y = numpify(x)
    assert isinstance(y, np.ndarray)
    assert x.shape == y.shape
    del x, y

    x = np.random.randn(5, 4)
    y = numpify(x)
    assert isinstance(y, np.ndarray)
    assert x.shape == y.shape
    del x, y

    x = np.random.randn(5, 4)
    y = numpify(x, dtype=np.float16)
    assert isinstance(y, np.ndarray)
    assert x.shape == y.shape
    assert y.dtype == np.float16
    del x, y

    # List
    x = [1, 2, 3, 4]
    y = numpify(x)
    assert isinstance(y, np.ndarray)
    assert np.allclose(x, y)
    del x, y

    x = [[1.2, 2.3], [3.4, 4.5]]
    y = numpify(x)
    assert isinstance(y, np.ndarray)
    assert np.allclose(x, y)
    del x, y

    # Tuple
    x = (1, 2, 3, 4)
    y = numpify(x)
    assert isinstance(y, np.ndarray)
    assert np.allclose(x, y)
    del x, y

    x = ((1.2, 2.3), (3.4, 4.5))
    y = numpify(x)
    assert isinstance(y, np.ndarray)
    assert np.allclose(x, y)
    del x, y

    # Scalar
    x = 1
    y = numpify(x)
    assert isinstance(y, np.ndarray)
    del x, y

    # Bool
    x = True
    y = numpify(x)
    assert isinstance(y, np.ndarray)
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
        print('ok')


def naive_primality(integer):
    r"""Naive way to test a prime by iterating over all preceding integers. """
    prime = True
    if integer <= 1:
        prime = False
    else:
        for i in range(2, integer):
            if integer % i == 0:
                prime = False

    return prime
    
    
class Worker(ProcessWorker):
    def work(self, task_id, task):
        return naive_primality(task)
    
    
class Master(ProcessMaster):
    def make_tasks(self):
        primes = [16127, 23251, 29611, 37199]
        non_primes = [5853, 7179, 6957]
        tasks = [primes[0], primes[1], non_primes[0], primes[2], non_primes[1], non_primes[2], primes[3]]
        
        return tasks


@pytest.mark.parametrize('num_worker', [1, 2, 3, 5, 7, 10])
def test_process_master_worker(num_worker):
    def check(master):
        assert all([not p.is_alive() for p in master.list_process])
        assert all([conn.closed for conn in master.master_conns])
        assert all([conn.closed for conn in master.worker_conns])
    
    master = Master(Worker, num_worker)
    assert master.num_worker == num_worker
    results = master()
    assert results == [True, True, False, True, False, False, True]
    check(master)
