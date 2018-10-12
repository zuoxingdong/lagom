import numpy as np
import pytest

from collections import OrderedDict

from lagom.envs.spaces import Box
from lagom.envs.spaces import Discrete
from lagom.envs.spaces import Dict
from lagom.envs.spaces import Tuple

import gym
from lagom.envs.spaces import convert_gym_space


def test_box():
    with pytest.raises(AssertionError):
        Box(-1.0, 1.0, dtype=None)
    with pytest.raises(AssertionError):
        Box(-1.0, [1.0, 2.0], np.float32, shape=(2,))
    with pytest.raises(AttributeError):
        Box(np.array([-1.0, -2.0]), [3.0, 4.0], np.float32)
    
    def check(box):
        assert all([dtype == np.float32 for dtype in [box.dtype, box.low.dtype, box.high.dtype]])
        assert all([s == (2, 3) for s in [box.shape, box.low.shape, box.high.shape]])
        assert np.allclose(box.low, np.full([2, 3], -1.0))
        assert np.allclose(box.high, np.full([2, 3], 1.0))
        sample = box.sample()
        assert sample.shape == (2, 3) and sample.dtype == np.float32
        assert box.flat_dim == 6 and isinstance(box.flat_dim, int)
        assert box.flatten(sample).shape == (6,)
        assert np.allclose(sample, box.unflatten(box.flatten(sample)))
        assert sample in box
        assert str(box) == 'Box(2, 3)'
        assert box == Box(-1.0, 1.0, np.float32, shape=[2, 3])
        del box, sample
    
    box1 = Box(-1.0, 1.0, np.float32, shape=[2, 3])
    check(box1)
    
    x = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    box2 = Box(low=-x, high=x, dtype=np.float32)
    check(box2)
    
    assert box1 == box2
    
    
def test_discrete():
    with pytest.raises(AssertionError):
        Discrete('no')
    with pytest.raises(AssertionError):
        Discrete(-1)
    
    discrete = Discrete(5)
    assert discrete.dtype == np.int32
    assert discrete.n == 5
    sample = discrete.sample()
    assert isinstance(sample, int)
    assert discrete.flat_dim == 5
    assert discrete.flatten(sample).shape == (5,)
    assert sample == discrete.unflatten(discrete.flatten(sample))
    assert sample in discrete
    assert discrete == Discrete(5)

    
def test_dict():
    with pytest.raises(AssertionError):
        Dict([Discrete(10), Box(-1, 1, np.float32, shape=(3,))])
        
    sensor_space = Dict({'position': Box(-100, 100, np.float32, shape=(3,)), 
                         'velocity': Box(-1, 1, np.float32, shape=(3,))})
    assert len(sensor_space.spaces) == 2
    assert 'position' in sensor_space.spaces and 'velocity' in sensor_space.spaces
    assert sensor_space.spaces['position'] == Box(-100, 100, np.float32, shape=(3,))
    assert sensor_space.spaces['velocity'] == Box(-1, 1, np.float32, shape=(3,))
    space = Dict({'sensors': sensor_space, 'score': Discrete(100)})
    assert len(space.spaces) == 2
    assert 'sensors' in space.spaces and 'score' in space.spaces
    assert space.spaces['sensors'] == sensor_space
    assert space.spaces['score'] == Discrete(100)
    sample = space.sample()
    assert isinstance(sample, dict) and len(sample) == 2
    assert isinstance(sample['sensors'], dict) and len(sample['sensors']) == 2
    assert sample['sensors'] in sensor_space
    assert sample['score'] in Discrete(100)
    assert space.flat_dim == 3+3+100
    assert space.flatten(sample).shape == (106,)
    sample2 = space.unflatten(space.flatten(sample))
    assert sample['score'] == sample2['score']
    assert np.allclose(sample['sensors']['position'], sample2['sensors']['position'])
    assert np.allclose(sample['sensors']['velocity'], sample2['sensors']['velocity'])
    assert sample in space
    
    
def test_tuple():
    with pytest.raises(AssertionError):
        Tuple(Discrete(10))
    
    space = Tuple([Discrete(5), 
                   Box(-1.0, 1.0, np.float32, shape=(2, 3)), 
                   Dict({'success': Discrete(2), 'velocity': Box(-1, 1, np.float32, shape=(1, 3))})])
    assert len(space.spaces) == 3
    assert space.spaces[0] == Discrete(5)
    assert space.spaces[1] == Box(-1.0, 1.0, np.float32, shape=(2, 3))
    assert space.spaces[2] == Dict({'success': Discrete(2), 'velocity': Box(-1, 1, np.float32, shape=(1, 3))})
    sample = space.sample()
    assert isinstance(sample, tuple) and len(sample) == 3
    assert sample[0] in Discrete(5)
    assert sample[1] in Box(-1.0, 1.0, np.float32, shape=(2, 3))
    assert sample[2] in Dict({'success': Discrete(2), 'velocity': Box(-1, 1, np.float32, shape=(1, 3))})
    assert space.flat_dim == 5+2*3+2+3
    assert space.flatten(sample).shape == (16,)
    sample2 = space.unflatten(space.flatten(sample))
    assert sample[0] == sample2[0]
    assert np.allclose(sample[1], sample2[1])
    assert sample[2]['success'] == sample2[2]['success']
    assert np.allclose(sample[2]['velocity'], sample2[2]['velocity'])
    sample in space

    
def test_convert_gym_space():
    # Discrete
    gym_space = gym.spaces.Discrete(n=5)
    lagom_space = convert_gym_space(gym_space)
    assert isinstance(lagom_space, Discrete)
    assert not isinstance(lagom_space, gym.spaces.Discrete)
    assert lagom_space.n == 5
    assert lagom_space.sample() in lagom_space

    del gym_space, lagom_space

    # Box
    gym_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(2, 3), dtype=np.float32)
    lagom_space = convert_gym_space(gym_space)
    assert isinstance(lagom_space, Box)
    assert not isinstance(lagom_space, gym.spaces.Box)
    assert lagom_space.shape == (2, 3)
    assert lagom_space.sample() in lagom_space

    del gym_space, lagom_space

    # Dict
    gym_space = gym.spaces.Dict({
        'sensors': gym.spaces.Dict({
            'position': gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32), 
            'velocity': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)}), 
        'charge': gym.spaces.Discrete(100)})
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

    del gym_space, lagom_space
    
    # Tuple
    gym_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Box(-1.0, 1.0, [2, 3], np.float32)))
    lagom_space = convert_gym_space(gym_space)
    assert isinstance(lagom_space, Tuple)
    assert not isinstance(lagom_space, gym.spaces.Tuple)
    assert len(lagom_space.spaces) == 2
    assert isinstance(lagom_space.spaces[0], Discrete)
    assert not isinstance(lagom_space.spaces[0], gym.spaces.Discrete)
    assert isinstance(lagom_space.spaces[1], Box)
    assert not isinstance(lagom_space.spaces[1], gym.spaces.Box)
    assert lagom_space.flat_dim == 2+2*3
    assert lagom_space.sample() in lagom_space

    del gym_space, lagom_space
