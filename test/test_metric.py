from copy import deepcopy
import pytest

import numpy as np
import torch

import gym

from lagom.agents import RandomAgent
from lagom.envs import make_vec_env
from lagom.runner import BatchHistory
from lagom.runner import EpisodeRunner
from lagom.runner import RollingSegmentRunner

from lagom.metric import returns
from lagom.metric import get_returns
from lagom.metric import bootstrapped_returns
from lagom.metric import get_bootstrapped_returns
from lagom.metric import td0_target
from lagom.metric import get_td0_target
from lagom.metric import td0_error
from lagom.metric import get_td0_error
from lagom.metric import gae
from lagom.metric import get_gae

from .sanity_env import SanityEnv


@pytest.mark.parametrize('num_env', [1, 3])
@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('mode', ['serial', 'parallel'])
@pytest.mark.parametrize('T', [1, 4, 6, 20])
@pytest.mark.parametrize('runner_class', [EpisodeRunner, RollingSegmentRunner])
def test_get_returns(num_env, init_seed, mode, T, runner_class):
    gamma = 0.1
    y1 = [0.1]
    y2 = [0.1 + gamma*0.2, 0.2]
    y3 = [0.1 + gamma*(0.2 + gamma*0.3), 
          0.2 + gamma*0.3, 
          0.3]
    y4 = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*0.4)), 
          0.2 + gamma*(0.3 + gamma*0.4), 
          0.3 + gamma*0.4, 
          0.4]
    y5 = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*0.5))), 
          0.2 + gamma*(0.3 + gamma*(0.4 + gamma*0.5)), 
          0.3 + gamma*(0.4 + gamma*0.5), 
          0.4 + gamma*0.5, 
          0.5]
    y6 = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*0.6)))), 
          0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*0.6))), 
          0.3 + gamma*(0.4 + gamma*(0.5 + gamma*0.6)), 
          0.4 + gamma*(0.5 + gamma*0.6), 
          0.5 + gamma*0.6, 
          0.6]
    ys = [None, y1, y2, y3, y4, y5, y6]

    make_env = lambda: SanityEnv()
    env = make_vec_env(make_env, num_env, init_seed, mode)
    agent = RandomAgent(None, env, None)
    runner = runner_class()
    D = runner(agent, env, T)
    Qs = get_returns(D, gamma)
    for n in range(D.N):
        out = []
        for t in D.Ts[n]:
            out += ys[t]
        assert np.allclose(out, Qs[n, :len(out)])
        
    # Raw test
    env = make_vec_env(make_env, 3, init_seed, mode)
    D = BatchHistory(env)
    D.done = [[[False, False, True]], 
              [[False, False, False, False, False]], 
              [[False, False, False, False, False, False, False, True]]]
    D.r = [[[1, 2, 3]], 
           [[1, 2, 3, 4, 5]], 
           [[1, 2, 3, 4, 5, 6, 7, 8]]]
    out = get_returns(D, 1.0)
    assert np.allclose(out[0], [6, 5, 3, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [15, 14, 12, 9, 5, 0, 0, 0])
    assert np.allclose(out[2], [36, 35, 33, 30, 26, 21, 15, 8]) 
    
    out = get_returns(D, 0.1)
    assert np.allclose(out[0], [1.23, 2.3, 3, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [1.2345, 2.345, 3.45, 4.5, 5, 0, 0, 0])
    assert np.allclose(out[2], [1.2345678, 2.345678, 3.45678, 4.5678, 5.678, 6.78, 7.8, 8])
    
    D = BatchHistory(env)
    D.done = [[[False, False, False, False, False]], 
              [[False, False, True], [False, False]], 
              [[True], [False, False, False, True]]]
    D.r = [[[1, 2, 3, 4, 5]], 
           [[1, 2, 3], [4, 5]], 
           [[1], [2, 3, 4, 5]]]
    out = get_returns(D, 1.0)
    assert np.allclose(out[0], [15, 14, 12, 9, 5])
    assert np.allclose(out[1], [6, 5, 3, 9, 5])
    assert np.allclose(out[2], [1, 14, 12, 9, 5])
    
    out = get_returns(D, 0.1)
    assert np.allclose(out[0], [1.2345, 2.345, 3.45, 4.5, 5])
    assert np.allclose(out[1], [1.23, 2.3, 3, 4.5, 5])
    assert np.allclose(out[2], [1, 2.345, 3.45, 4.5, 5])


@pytest.mark.parametrize('gamma', [0.1, 0.5])
def test_get_bootstrapped_returns(gamma):
    np.random.seed(3)

    make_env = lambda: SanityEnv()
    env = make_vec_env(make_env, 3, 0, 'serial')
    agent = RandomAgent(None, env, None)
    runner = EpisodeRunner()
    D = runner(agent, env, 5)
    last_Vs = D.last_observations
    Qs = get_bootstrapped_returns(D, last_Vs.tolist(), gamma)

    assert D.done[0][0] == [False, False, False, False, False]
    assert D.done[1][0] == [False, False, False, True]
    assert D.done[2][0] == [False, False, False, False, True]

    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_Vs[0])))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_Vs[0]))), 
         0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_Vs[0])), 
         0.4 + gamma*(0.5 + gamma*last_Vs[0]), 
         0.5 + gamma*last_Vs[0]]
    assert np.allclose(y, Qs[0, :len(y)])
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*last_Vs[1]*0.0))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*last_Vs[1]*0.0)), 
         0.3 + gamma*(0.4 + gamma*last_Vs[1]*0.0), 
         0.4 + gamma*last_Vs[1]*0.0]
    assert np.allclose(y, Qs[1, :len(y)])
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_Vs[2]*0.0)))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_Vs[2]*0.0))), 
         0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_Vs[2]*0.0)), 
         0.4 + gamma*(0.5 + gamma*last_Vs[2]*0.0),
         0.5 + gamma*last_Vs[2]*0.0]
    assert np.allclose(y, Qs[2, :len(y)])

    runner = EpisodeRunner()
    D = runner(agent, env, 3)
    last_Vs = D.last_observations
    Qs = get_bootstrapped_returns(D, last_Vs.tolist(), gamma)

    assert D.done[0][0] == [False, False, False]
    assert D.done[1][0] == [False, False, False]
    assert D.done[2][0] == [False, False, False]

    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*last_Vs[1])), 
         0.2 + gamma*(0.3 + gamma*last_Vs[1]), 
         0.3 + gamma*last_Vs[1]]
    assert np.allclose(y, Qs[0, :len(y)])
    assert np.allclose(y, Qs[1, :len(y)])
    assert np.allclose(y, Qs[2, :len(y)])

    # rolling segment runner
    runner = RollingSegmentRunner()
    D = runner(agent, env, 8, reset=True)
    last_Vs = D.last_observations
    Qs = get_bootstrapped_returns(D, last_Vs.tolist(), gamma)

    assert D.done[0][0] == [False, False, False, False, False, True]
    assert D.done[0][1] == [False, False]
    assert D.done[1][0] == [False, False, False, True]
    assert D.done[1][1] == [False, False, False, True]
    assert D.done[2][0] == [False, False, False, False, True]
    assert D.done[2][1] == [False, False, False]

    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*(0.6 + gamma*last_Vs[0]*0.0))))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*(0.6 + gamma*last_Vs[0]*0.0)))), 
         0.3 + gamma*(0.4 + gamma*(0.5 + gamma*(0.6 + gamma*last_Vs[0]*0.0))), 
         0.4 + gamma*(0.5 + gamma*(0.6 + gamma*last_Vs[0]*0.0)), 
         0.5 + gamma*(0.6 + gamma*last_Vs[0]*0.0), 
         0.6 + gamma*last_Vs[0]*0.0]
    assert np.allclose(y, Qs[0, :6])
    y = [0.1 + gamma*(0.2 + gamma*last_Vs[1]), 
         0.2 + gamma*last_Vs[1]]
    assert np.allclose(y, Qs[0, 6:])
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*last_Vs[2]*0.0))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*last_Vs[2]*0.0)), 
         0.3 + gamma*(0.4 + gamma*last_Vs[2]*0.0), 
         0.4 + gamma*last_Vs[2]*0.0]
    assert np.allclose(y, Qs[1, :4])
    assert np.allclose(y, Qs[1, 4:])
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_Vs[4]*0.0)))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_Vs[4]*0.0))), 
         0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_Vs[4]*0.0)), 
         0.4 + gamma*(0.5 + gamma*last_Vs[4]*0.0), 
         0.5 + gamma*last_Vs[4]*0.0]
    assert np.allclose(y, Qs[2, :5])
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*last_Vs[-1])), 
         0.2 + gamma*(0.3 + gamma*last_Vs[-1]), 
         0.3 + gamma*last_Vs[-1]]
    assert np.allclose(y, Qs[2, 5:])
    
    # Raw test
    D = BatchHistory(env)
    D.done = [[[False, False, True]], 
              [[False, False, False, False, False]], 
              [[False, False, False, False, False, False, False, True]]]
    D.r = [[[1, 2, 3]], 
           [[1, 2, 3, 4, 5]], 
           [[1, 2, 3, 4, 5, 6, 7, 8]]]
    last_Vs = [10, 20, 30]
    out = get_bootstrapped_returns(D, deepcopy(last_Vs), 1.0)
    assert np.allclose(out[0], [6, 5, 3, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [35, 34, 32, 29, 25, 0, 0, 0])
    assert np.allclose(out[2], [36, 35, 33, 30, 26, 21, 15, 8])
    
    out = get_bootstrapped_returns(D, deepcopy(last_Vs), 0.1)
    assert np.allclose(out[0], [1.23, 2.3, 3, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [1.2347, 2.347, 3.47, 4.7, 7, 0, 0, 0])
    assert np.allclose(out[2], [1.2345678, 2.345678, 3.45678, 4.5678, 5.678, 6.78, 7.8, 8])
    
    D = BatchHistory(env)
    D.done = [[[False, False, False, False, False]], 
              [[False, False, True], [False, False]], 
              [[True], [False, False, False, True]]]
    D.r = [[[1, 2, 3, 4, 5]], 
           [[1, 2, 3], [4, 5]], 
           [[1], [2, 3, 4, 5]]]
    last_Vs = [10, 99, 20, 99, 30]
    out = get_bootstrapped_returns(D, deepcopy(last_Vs), 1.0)
    assert np.allclose(out[0], [25, 24, 22, 19, 15])
    assert np.allclose(out[1], [6, 5, 3, 29, 25])
    assert np.allclose(out[2], [1, 14, 12, 9, 5])
    
    out = get_bootstrapped_returns(D, deepcopy(last_Vs), 0.1)
    assert np.allclose(out[0], [1.2346, 2.346, 3.46, 4.6, 6])
    assert np.allclose(out[1], [1.23, 2.3, 3, 4.7, 7])
    assert np.allclose(out[2], [1, 2.345, 3.45, 4.5, 5])
    


@pytest.mark.parametrize('gamma', [0.1, 0.5])
def test_get_td0_target(gamma):
    make_env = lambda: SanityEnv()
    env = make_vec_env(make_env, 3, 0, 'serial')

    # Episode: cut-off 
    D = BatchHistory(env)
    D.done = [[[False, False, False, True]], 
              [[False, False, False, False]], 
              [[False, False, False, False]]]
    D.r = [[[0.1, 0.2, 0.3, 0.4]], 
           [[0.1, 0.2, 0.3, 0.4]], 
           [[0.1, 0.2, 0.3, 0.4]]]
    Vs = [[1, 2, 3, 4], 
          [1, 2, 3, 4], 
          [1, 2, 3, 4]]
    last_Vs = [40, 40, 40]
    td0 = get_td0_target(D, Vs, last_Vs, gamma)

    y = [0.1 + gamma*2, 
         0.2 + gamma*3,
         0.3 + gamma*4, 
         0.4 + gamma*40*0.0]
    assert np.allclose(y, td0[0])
    y = [0.1 + gamma*2, 
         0.2 + gamma*3, 
         0.3 + gamma*4,
         0.4 + gamma*40]
    assert np.allclose(y, td0[1])
    y = [0.1 + gamma*2, 
         0.2 + gamma*3, 
         0.3 + gamma*4,
         0.4 + gamma*40]
    assert np.allclose(y, td0[2])

    # Episode: longer timesteps
    D = BatchHistory(env)
    D.done = [[[False, False, False, True]],
              [[False, False, False, False, False, True]],
              [[False, False, False, False, True]]]
    D.r = [[[0.1, 0.2, 0.3, 0.4]], 
           [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], 
           [[0.1, 0.2, 0.3, 0.4, 0.5]]]
    Vs = [[1, 2, 3, 4, 10, 10], 
          [1, 2, 3, 4, 5, 6], 
          [1, 2, 3, 4, 5, 10]]
    last_Vs = [40., 60, 50]
    td0 = get_td0_target(D, Vs, last_Vs, gamma)

    y = [0.1 + gamma*2, 
         0.2 + gamma*3, 
         0.3 + gamma*4, 
         0.4 + gamma*40*0.0, 
         0.0, 
         0.0]
    assert np.allclose(y, td0[0])
    y = [0.1 + gamma*2, 
         0.2 + gamma*3, 
         0.3 + gamma*4, 
         0.4 + gamma*5, 
         0.5 + gamma*6, 
         0.6 + gamma*60*0.0]
    assert np.allclose(y, td0[1])
    y = [0.1 + gamma*2, 
         0.2 + gamma*3,  
         0.3 + gamma*4, 
         0.4 + gamma*5, 
         0.5 + gamma*50*0.0, 
         0.0]
    assert np.allclose(y, td0[2])

    # Rolling segment
    D = BatchHistory(env)
    D.done = [[[False, False, False, False, False, True], [False]], 
              [[False, False, False, False, False, True], [False]], 
              [[False, False, False, False, True], [False, False]]]
    D.r = [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1]], 
           [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1]], 
           [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2]]]
    Vs = [[1, 2, 3, 4, 5, 6, 1], 
          [1, 2, 3, 4, 5, 6, 1], 
          [1, 2, 3, 4, 5, 1, 2]]
    last_Vs = [60, 10, 60, 10, 50, 20]
    td0 = get_td0_target(D, Vs, last_Vs, gamma)

    y = [0.1 + gamma*2, 
         0.2 + gamma*3, 
         0.3 + gamma*4, 
         0.4 + gamma*5, 
         0.5 + gamma*6, 
         0.6 + gamma*60*0.0, 
         0.1 + gamma*10]
    assert np.allclose(y, td0[0])
    assert np.allclose(y, td0[1])
    y = [0.1 + gamma*2, 
         0.2 + gamma*3,
         0.3 + gamma*4,
         0.4 + gamma*5,
         0.5 + gamma*50*0.0, 
         0.1 + gamma*2, 
         0.2 + gamma*20]
    assert np.allclose(y, td0[2])
    
    # Raw test
    D = BatchHistory(env)
    D.done = [[[False, False, True]], 
              [[False, False, False, False, False]], 
              [[False, False, False, False, False, False, False, True]]]
    D.r = [[[1, 2, 3]], 
           [[1, 2, 3, 4, 5]], 
           [[1, 2, 3, 4, 5, 6, 7, 8]]]
    Vs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1], 
          [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], 
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    last_Vs = [10, 20, 30]
    out = get_td0_target(D, deepcopy(Vs), deepcopy(last_Vs), 1.0)
    assert np.allclose(out[0], [2.1, 4.1, 3, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [2.5, 4.5, 6.5, 8.5, 25, 0, 0, 0])
    assert np.allclose(out[2], [3, 5, 7, 9, 11, 13, 15, 8])
    
    out = get_td0_target(D, deepcopy(Vs), deepcopy(last_Vs), 0.1)
    assert np.allclose(out[0], [1.11, 2.21, 3, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [1.15, 2.25, 3.35, 4.45, 7, 0, 0, 0])
    assert np.allclose(out[2], [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8])
    
    # segment
    D = BatchHistory(env)
    D.done = [[[False, False, False, False, False]], 
              [[False, False, True], [False, False]], 
              [[True], [False, False, False, True]]]
    D.r = [[[1, 2, 3, 4, 5]], 
           [[1, 2, 3], [4, 5]], 
           [[1], [2, 3, 4, 5]]]
    Vs = [[0.1, 1.1, 2.1, 3.1, 4.1], 
          [0.5, 1.5, 2.5, 3.5, 4.5], 
          [1.0, 2.0, 3.0, 4.0, 5.0]]
    last_Vs = [10, 99, 20, 99, 30]
    out = get_td0_target(D, deepcopy(Vs), deepcopy(last_Vs), 1.0)
    assert np.allclose(out[0], [2.1, 4.1, 6.1, 8.1, 15])
    assert np.allclose(out[1], [2.5, 4.5, 3, 8.5, 25])
    assert np.allclose(out[2], [1, 5, 7, 9, 5])
    
    out = get_td0_target(D, deepcopy(Vs), deepcopy(last_Vs), 0.1)
    assert np.allclose(out[0], [1.11, 2.21, 3.31, 4.41, 6])
    assert np.allclose(out[1], [1.15, 2.25, 3, 4.45, 7])
    assert np.allclose(out[2], [1, 2.3, 3.4, 4.5, 5])


@pytest.mark.parametrize('gamma', [0.1, 0.5])
def test_get_td0_error(gamma):
    make_env = lambda: SanityEnv()
    env = make_vec_env(make_env, 3, 0, 'serial')

    # Episode: cut-off 
    D = BatchHistory(env)
    D.done = [[[False, False, False, True]], 
              [[False, False, False, False]], 
              [[False, False, False, False]]]
    D.r = [[[0.1, 0.2, 0.3, 0.4]], 
           [[0.1, 0.2, 0.3, 0.4]], 
           [[0.1, 0.2, 0.3, 0.4]]]
    Vs = [[1, 2, 3, 4], 
          [1, 2, 3, 4], 
          [1, 2, 3, 4]]
    last_Vs = [40, 40, 40]
    td0 = get_td0_error(D, Vs, last_Vs, gamma)

    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2,
         0.3 + gamma*4 - 3, 
         0.4 + gamma*40*0.0 - 4]
    assert np.allclose(y, td0[0])
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2, 
         0.3 + gamma*4 - 3,
         0.4 + gamma*40 - 4]
    assert np.allclose(y, td0[1])
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2, 
         0.3 + gamma*4 - 3,
         0.4 + gamma*40 - 4]
    assert np.allclose(y, td0[2])

    # Episode: longer timesteps
    D = BatchHistory(env)
    D.done = [[[False, False, False, True]],
              [[False, False, False, False, False, True]],
              [[False, False, False, False, True]]]
    D.r = [[[0.1, 0.2, 0.3, 0.4]], 
           [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], 
           [[0.1, 0.2, 0.3, 0.4, 0.5]]]
    Vs = [[1, 2, 3, 4, 10, 10], 
          [1, 2, 3, 4, 5, 6], 
          [1, 2, 3, 4, 5, 10]]
    last_Vs = [40., 60, 50]
    td0 = get_td0_error(D, Vs, last_Vs, gamma)

    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2, 
         0.3 + gamma*4 - 3, 
         0.4 + gamma*40*0.0 - 4, 
         0.0, 
         0.0]
    assert np.allclose(y, td0[0])
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2, 
         0.3 + gamma*4 - 3, 
         0.4 + gamma*5 - 4, 
         0.5 + gamma*6 - 5,  
         0.6 + gamma*60*0.0 - 6]
    assert np.allclose(y, td0[1])
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2,  
         0.3 + gamma*4 - 3, 
         0.4 + gamma*5 - 4, 
         0.5 + gamma*50*0.0 - 5, 
         0.0]
    assert np.allclose(y, td0[2])

    # Rolling segment
    D = BatchHistory(env)
    D.done = [[[False, False, False, False, False, True], [False]], 
              [[False, False, False, False, False, True], [False]], 
              [[False, False, False, False, True], [False, False]]]
    D.r = [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1]], 
           [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1]], 
           [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2]]]
    Vs = [[1, 2, 3, 4, 5, 6, 1], 
          [1, 2, 3, 4, 5, 6, 1], 
          [1, 2, 3, 4, 5, 1, 2]]
    last_Vs = [60, 10, 60, 10, 50, 20]
    td0 = get_td0_error(D, Vs, last_Vs, gamma)

    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2, 
         0.3 + gamma*4 - 3, 
         0.4 + gamma*5 - 4, 
         0.5 + gamma*6 - 5, 
         0.6 + gamma*60*0.0 - 6, 
         0.1 + gamma*10 - 1]
    assert np.allclose(y, td0[0])
    assert np.allclose(y, td0[1])
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2,
         0.3 + gamma*4 - 3,
         0.4 + gamma*5 - 4,
         0.5 + gamma*50*0.0 - 5, 
         0.1 + gamma*2 - 1, 
         0.2 + gamma*20 - 2]
    assert np.allclose(y, td0[2])
    
    # raw test
    D = BatchHistory(env)
    D.done = [[[False, False, True]], 
              [[False, False, False, False, False]], 
              [[False, False, False, False, False, False, False, True]]]
    D.r = [[[1, 2, 3]], 
           [[1, 2, 3, 4, 5]], 
           [[1, 2, 3, 4, 5, 6, 7, 8]]]
    Vs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1], 
          [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], 
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    last_Vs = [10, 20, 30]
    out = get_td0_error(D, deepcopy(Vs), deepcopy(last_Vs), 1.0)
    assert np.allclose(out[0], [2.0, 3, 0.9, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [2, 3, 4, 5, 20.5, 0, 0, 0])
    assert np.allclose(out[2], [2, 3, 4, 5, 6, 7, 8, 0])
    
    out = get_td0_error(D, deepcopy(Vs), deepcopy(last_Vs), 0.1)
    assert np.allclose(out[0], [1.01, 1.11, 0.9, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [0.65, 0.75, 0.85, 0.95, 2.5, 0, 0, 0])
    assert np.allclose(out[2], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0])
    
    # segment
    D = BatchHistory(env)
    D.done = [[[False, False, False, False, False]], 
              [[False, False, True], [False, False]], 
              [[True], [False, False, False, True]]]
    D.r = [[[1, 2, 3, 4, 5]], 
           [[1, 2, 3], [4, 5]], 
           [[1], [2, 3, 4, 5]]]
    Vs = [[0.1, 1.1, 2.1, 3.1, 4.1], 
          [0.5, 1.5, 2.5, 3.5, 4.5], 
          [1.0, 2.0, 3.0, 4.0, 5.0]]
    last_Vs = [10, 99, 20, 99, 30]
    out = out = get_td0_error(D, deepcopy(Vs), deepcopy(last_Vs), 1.0)
    assert np.allclose(out[0], [2.0, 3, 4, 5, 10.9])
    assert np.allclose(out[1], [2, 3, 0.5, 5, 20.5])
    assert np.allclose(out[2], [0, 3, 4, 5, 0])
    
    out = out = get_td0_error(D, deepcopy(Vs), deepcopy(last_Vs), 0.1)
    assert np.allclose(out[0], [1.01, 1.11, 1.21, 1.31, 1.9])
    assert np.allclose(out[1], [0.65, 0.75, 0.5, 0.95, 2.5])
    assert np.allclose(out[2], [0, 0.3, 0.4, 0.5, 0])


def test_get_gae():
    make_env = lambda: SanityEnv()
    env = make_vec_env(make_env, 3, 0, 'serial')

    # Episode
    D = BatchHistory(env)
    D.done = [[[False, False, True]], 
              [[False, False, False, False, False]], 
              [[False, False, False, False, False, False, False, True]]]
    D.r = [[[1, 2, 3]], 
           [[1, 2, 3, 4, 5]],
           [[1, 2, 3, 4, 5, 6, 7, 8]]]
    Vs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1], 
          [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], 
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    last_Vs = [10, 20, 30]

    out = get_gae(D, deepcopy(Vs), deepcopy(last_Vs), 1.0, 0.5)
    assert np.allclose(out[0], [3.725, 3.45, 0.9, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [6.40625, 8.8125, 11.625, 15.25, 20.5, 0, 0, 0])
    assert np.allclose(out[2], [5.84375, 7.6875, 9.375, 10.75, 11.5, 11., 8, 0.])

    out = get_gae(D, deepcopy(Vs), deepcopy(last_Vs), 0.1, 0.2)
    assert np.allclose(out[0], [1.03256, 1.128, 0.9, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [0.665348, 0.7674, 0.87, 1, 2.5, 0, 0, 0])
    assert np.allclose(out[2], [0.206164098, 0.308204915, 0.410245728, 0.5122864, 0.61432, 0.716, 0.8, 0])

    # Segment
    D = BatchHistory(env)
    D.done = [[[False, False, False, False, False]], 
              [[False, False, True], [False, False]], 
              [[True], [False, False, False, True]]]
    D.r = [[[1, 2, 3, 4, 5]], 
           [[1, 2, 3], [4, 5]], 
           [[1], [2, 3, 4, 5]]]
    Vs = [[0.1, 1.1, 2.1, 3.1, 4.1], 
          [0.5, 1.5, 2.5, 3.5, 4.5], 
          [1.0, 2.0, 3.0, 4.0, 5.0]]
    last_Vs = [10, 99, 20, 99, 30]

    out = get_gae(D, deepcopy(Vs), deepcopy(last_Vs), 1.0, 0.5)
    assert np.allclose(out[0], [5.80625, 7.6125, 9.225, 10.45, 10.9])
    assert np.allclose(out[1], [3.625, 3.25, 0.5, 15.25, 20.5])
    assert np.allclose(out[2], [0, 6.25, 6.5, 5, 0])

    out = get_gae(D, deepcopy(Vs), deepcopy(last_Vs), 0.1, 0.2)
    assert np.allclose(out[0], [1.03269478, 1.1347393, 1.23696, 1.348, 1.9])
    assert np.allclose(out[1], [0.6652, 0.76, 0.5, 1, 2.5])
    assert np.allclose(out[2], [0, 0.3082, 0.41, 0.5, 0])
    
    # raw test
    D = BatchHistory(env)
    D.done = [[[False, False, True]], 
              [[False, False, False, False, False]], 
              [[False, False, False, False, False, False, False, True]]]
    D.r = [[[1, 2, 3]], 
           [[1, 2, 3, 4, 5]], 
           [[1, 2, 3, 4, 5, 6, 7, 8]]]
    Vs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1], 
          [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], 
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    last_Vs = [10, 20, 30]
    out = get_gae(D, deepcopy(Vs), deepcopy(last_Vs), 1.0, 0.5)
    assert np.allclose(out[0], [3.725, 3.45, 0.9, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [6.40625, 8.8125, 11.625, 15.25, 20.5, 0, 0, 0])
    assert np.allclose(out[2], [5.84375, 7.6875, 9.375, 10.75, 11.5, 11., 8, 0.])
    
    out = get_gae(D, deepcopy(Vs), deepcopy(last_Vs), 0.1, 0.2)
    assert np.allclose(out[0], [1.03256, 1.128, 0.9, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [0.665348, 0.7674, 0.87, 1, 2.5, 0, 0, 0])
    assert np.allclose(out[2], [0.206164098, 0.308204915, 0.410245728, 0.5122864, 0.61432, 0.716, 0.8, 0])
    
    # segment
    D = BatchHistory(env)
    D.done = [[[False, False, False, False, False]], 
              [[False, False, True], [False, False]], 
              [[True], [False, False, False, True]]]
    D.r = [[[1, 2, 3, 4, 5]], 
           [[1, 2, 3], [4, 5]], 
           [[1], [2, 3, 4, 5]]]
    Vs = [[0.1, 1.1, 2.1, 3.1, 4.1], 
          [0.5, 1.5, 2.5, 3.5, 4.5], 
          [1.0, 2.0, 3.0, 4.0, 5.0]]
    last_Vs = [10, 99, 20, 99, 30]
    out = get_gae(D, deepcopy(Vs), deepcopy(last_Vs), 1.0, 0.5)
    assert np.allclose(out[0], [5.80625, 7.6125, 9.225, 10.45, 10.9])
    assert np.allclose(out[1], [3.625, 3.25, 0.5, 15.25, 20.5])
    assert np.allclose(out[2], [0, 6.25, 6.5, 5, 0])
    
    out = get_gae(D, deepcopy(Vs), deepcopy(last_Vs), 0.1, 0.2)
    assert np.allclose(out[0], [1.03269478, 1.1347393, 1.23696, 1.348, 1.9])
    assert np.allclose(out[1], [0.6652, 0.76, 0.5, 1, 2.5])
    assert np.allclose(out[2], [0, 0.3082, 0.41, 0.5, 0])
