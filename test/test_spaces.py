import numpy as np
import pytest

from collections import OrderedDict

from lagom.envs.spaces import Box
from lagom.envs.spaces import Discrete
from lagom.envs.spaces import Product
from lagom.envs.spaces import Dict

import gym
from lagom.envs.spaces import convert_gym_space


def test_box():
    with pytest.raises(AssertionError):
        Box(-1.0, 1.0, dtype=None)
    box1 = Box(-1.0, 1.0, shape=[2, 3], dtype=np.float32)
    assert box1.dtype == np.float32
    assert box1.shape == (2, 3)
    assert box1.low.shape == (2, 3)
    assert box1.high.shape == (2, 3)
    s1 = box1.sample()
    assert s1.shape == (2, 3)
    assert s1 in box1

    assert box1.flat_dim == 6
    assert box1.flatten(s1).shape == (6,)
    assert np.allclose(s1, box1.unflatten(box1.flatten(s1)))

    low = np.array([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])
    box2 = Box(low=low, high=-low, dtype=np.float32)
    assert box2.dtype == np.float32
    assert box2.shape == (2, 3)
    assert box2.low.shape == (2, 3)
    assert box2.high.shape == (2, 3)
    s2 = box2.sample()
    assert s2.shape == (2, 3)
    assert s2 in box2

    assert box1 == box2

    assert box2.flat_dim == 6
    assert box2.flatten(s2).shape == (6,)
    assert np.allclose(s2, box2.unflatten(box2.flatten(s2)))
    
def test_discrete():
    with pytest.raises(AssertionError):
        Discrete('no')
    with pytest.raises(AssertionError):
        Discrete(-1)
    
    discrete = Discrete(5)
    assert discrete.dtype == np.int32
    assert discrete.n == 5
    assert discrete.shape is None
    s1 = discrete.sample()
    assert isinstance(s1, int)
    assert s1 in discrete
    
    assert discrete.flat_dim == 5
    assert discrete.flatten(s1).shape == (5,)
    assert isinstance(discrete.unflatten(discrete.flatten(s1)), int)
    
    discrete2 = Discrete(5)
    assert discrete == discrete2
    
def test_product():
    box = Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32)
    discrete = Discrete(5)
    product = Product([discrete, box])
    assert product.dtype is None
    assert product.shape is None
    assert isinstance(product.spaces, tuple)
    assert len(product.spaces) == 2

    s1 = product.sample()
    assert isinstance(s1, tuple)
    assert len(s1) == 2
    assert isinstance(s1[0], int)
    assert isinstance(s1[1], np.ndarray)
    assert s1[1].shape == (2, 3)
    assert s1 in product

    assert product.flat_dim == 2*3 + 5
    assert isinstance(product.flatten(s1), np.ndarray)
    assert product.flatten(s1).shape == (11,)
    s2 = product.unflatten(product.flatten(s1))
    assert isinstance(s2, tuple)
    assert len(s2) == 2
    assert isinstance(s2[0], int)
    assert isinstance(s2[1], np.ndarray)
    assert s2[1].shape == (2, 3)

def test_dict():
    # Simple use
    space = Dict({'position': Discrete(2), 'velocity': Box(low=-1.0, high=1.0, shape=(1, 2), dtype=np.float32)})

    assert space.dtype is None
    assert space.shape is None
    assert isinstance(space.spaces, dict)
    assert len(space.spaces) == 2
    assert 'position' in space.spaces
    assert 'velocity' in space.spaces
    assert isinstance(space.spaces['position'], Discrete)
    assert isinstance(space.spaces['velocity'], Box)
    assert space.spaces['position'].n == 2
    assert space.spaces['velocity'].shape == (1, 2)

    s1 = space.sample()
    assert s1 in space
    assert isinstance(s1, OrderedDict)
    assert len(s1) == 2
    assert 'position' in s1
    assert 'velocity' in s1
    assert isinstance(s1['position'], int)
    assert isinstance(s1['velocity'], np.ndarray)
    assert s1['position'] in space.spaces['position']
    assert s1['velocity'] in space.spaces['velocity']

    assert space.flat_dim == 2+1*2
    flat_s1 = space.flatten(s1)
    assert isinstance(flat_s1, np.ndarray)
    assert flat_s1.dtype == np.float32
    assert flat_s1.shape == (4,)
    assert flat_s1[:2].sum() == 1
    assert flat_s1.max() <= 1.0
    assert flat_s1.min() >= -1.0
    unflat_s1 = space.unflatten(flat_s1)
    assert isinstance(unflat_s1, OrderedDict)
    assert len(unflat_s1) == 2
    assert 'position' in unflat_s1
    assert 'velocity' in unflat_s1
    assert unflat_s1['position'] == s1['position']
    assert np.allclose(unflat_s1['velocity'], s1['velocity'])
    assert unflat_s1 in space

    del space
    del s1
    del flat_s1
    del unflat_s1

    # Nested
    space = Dict({
        'sensors': Dict({
            'position': Box(-100, 100, shape=(3,), dtype=np.float32), 
            'velocity': Box(-1, 1, shape=(3,), dtype=np.float32)
            }),
        'score': Discrete(100)  
    })
    # Do not do redundant tests
    assert space.flat_dim == 3+3+100
    s1 = space.sample()
    assert s1 in space
    assert 'score' in s1
    assert 'sensors' in s1
    assert 'position' in s1['sensors']
    assert 'velocity' in s1['sensors']
    flat_s1 = space.flatten(s1)
    assert flat_s1.shape == (3+3+100,)
    unflat_s1 = space.unflatten(flat_s1)
    assert unflat_s1 in space

def test_convert_gym_space():
    # Discrete
    gym_space = gym.spaces.Discrete(n=5)
    assert isinstance(gym_space, gym.spaces.Discrete)
    lagom_space = convert_gym_space(gym_space)
    assert isinstance(lagom_space, Discrete)
    assert not isinstance(lagom_space, gym.spaces.Discrete)
    assert lagom_space.n == 5
    assert lagom_space.sample() in lagom_space

    del gym_space
    del lagom_space

    # Box
    gym_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(2, 3), dtype=np.float32)
    assert isinstance(gym_space, gym.spaces.Box)
    lagom_space = convert_gym_space(gym_space)
    assert isinstance(lagom_space, Box)
    assert not isinstance(lagom_space, gym.spaces.Box)
    assert lagom_space.shape == (2, 3)
    assert lagom_space.sample() in lagom_space

    del gym_space
    del lagom_space

    # Tuple - Product
    gym_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Box(-1.0, 1.0, [2, 3], np.float32)))
    assert isinstance(gym_space, gym.spaces.Tuple)
    lagom_space = convert_gym_space(gym_space)
    assert isinstance(lagom_space, Product)
    assert not isinstance(lagom_space, gym.spaces.Tuple)
    assert len(lagom_space.spaces) == 2
    assert isinstance(lagom_space.spaces[0], Discrete)
    assert not isinstance(lagom_space.spaces[0], gym.spaces.Discrete)
    assert isinstance(lagom_space.spaces[1], Box)
    assert not isinstance(lagom_space.spaces[1], gym.spaces.Box)
    assert lagom_space.flat_dim == 2+2*3
    assert lagom_space.sample() in lagom_space

    del gym_space
    del lagom_space

    # Dict
    gym_space = gym.spaces.Dict({
        'sensors': gym.spaces.Dict({
            'position': gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32), 
            'velocity': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)}), 
        'charge': gym.spaces.Discrete(100)})
    assert isinstance(gym_space, gym.spaces.Dict)
    lagom_space = convert_gym_space(gym_space)
    assert isinstance(lagom_space, Dict)
    assert not isinstance(lagom_space, gym.spaces.Dict)
    assert len(lagom_space.spaces) == 2
    assert isinstance(lagom_space.spaces['sensors'], Dict)
    assert len(lagom_space.spaces['sensors'].spaces) == 2
    assert isinstance(lagom_space.spaces['charge'], Discrete)
    assert isinstance(lagom_space.spaces['sensors'].spaces['velocity'], Box)
    assert lagom_space.flat_dim == 100+3+3
    assert lagom_space.sample() in lagom_space

    del gym_space
    del lagom_space
    