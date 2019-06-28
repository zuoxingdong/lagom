from copy import deepcopy
import pytest

import numpy as np
import torch

import gym
from gym.wrappers import TimeLimit

from lagom import RandomAgent
from lagom import EpisodeRunner
from lagom.envs import make_vec_env
from lagom.envs.wrappers import StepInfo
from lagom.envs.wrappers import VecStepInfo

from lagom.metric import Trajectory
from lagom.metric import returns
from lagom.metric import bootstrapped_returns
from lagom.metric import td0_target
from lagom.metric import td0_error
from lagom.metric import gae

from .sanity_env import SanityEnv


@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('T', [1, 5, 100])
def test_trajectory(init_seed, T):
    make_env = lambda: TimeLimit(SanityEnv())
    env = make_vec_env(make_env, 1, init_seed)  # single environment
    env = VecStepInfo(env)
    D = Trajectory()
    assert len(D) == 0
    assert not D.completed
    
    observation, _ = env.reset()
    D.add_observation(observation)
    for t in range(T):
        action = [env.action_space.sample()]
        next_observation, reward, [step_info] = env.step(action)
        if step_info.last:
            D.add_observation([step_info['last_observation']])
        else:
            D.add_observation(next_observation)
        D.add_action(action)
        D.add_reward(reward)
        D.add_step_info(step_info)
        observation = next_observation
        if step_info.last:
            with pytest.raises(AssertionError):
                D.add_observation(observation)
            break
    assert len(D) > 0
    assert len(D) <= T
    assert len(D) + 1 == len(D.observations)
    assert len(D) + 1 == len(D.numpy_observations)
    assert len(D) == len(D.actions)
    assert len(D) == len(D.numpy_actions)
    assert len(D) == len(D.rewards)
    assert len(D) == len(D.numpy_rewards)
    assert len(D) == len(D.numpy_dones)
    assert len(D) == len(D.numpy_masks)
    assert np.allclose(np.logical_not(D.numpy_dones), D.numpy_masks)
    assert len(D) == len(D.step_infos)
    if len(D) < T:
        assert step_info.last
        assert D.completed
        assert D.reach_terminal
        assert not D.reach_time_limit
        assert np.allclose(D.observations[-1], [step_info['last_observation']])
    if not step_info.last:
        assert not D.completed
        assert not D.reach_terminal
        assert not D.reach_time_limit


@pytest.mark.parametrize('num_env', [1])
@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('T', [1, 4, 6, 20])
def test_returns(num_env, init_seed, T):
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

    make_env = lambda: TimeLimit(SanityEnv())
    env = make_vec_env(make_env, num_env, init_seed)
    env = VecStepInfo(env)
    agent = RandomAgent(None, env, None)
    runner = EpisodeRunner()
    D = runner(agent, env, T)
    for traj in D:
        Qs = returns(gamma, traj)
        assert np.allclose(ys[len(traj)], Qs)
    
    # Raw test
    D = Trajectory()
    D.dones = [False, False, True]
    D.rewards = [1, 2, 3]
    out = returns(1.0, D)
    assert np.allclose(out, [6, 5, 3])
    out = returns(0.1, D)
    assert np.allclose(out, [1.23, 2.3, 3])
    
    D = Trajectory()
    D.dones = [False, False, False, False, False]
    D.rewards = [1, 2, 3, 4, 5]
    out = returns(1.0, D)
    assert np.allclose(out, [15, 14, 12, 9, 5])
    out = returns(0.1, D)
    assert np.allclose(out, [1.2345, 2.345, 3.45, 4.5, 5])
    
    D = Trajectory()
    D.dones = [False, False, False, False, False, False, False, True]
    D.rewards = [1, 2, 3, 4, 5, 6, 7, 8]
    out = returns(1.0, D)
    assert np.allclose(out, [36, 35, 33, 30, 26, 21, 15, 8])
    out = returns(0.1, D)
    assert np.allclose(out, [1.2345678, 2.345678, 3.45678, 4.5678, 5.678, 6.78, 7.8, 8])
    
    
@pytest.mark.parametrize('gamma', [0.1, 0.5])
@pytest.mark.parametrize('last_V', [0.0, 2.0])
def test_bootstrapped_returns(gamma, last_V):
    D = Trajectory()
    dones = [False, False, False, False, False]
    infos = [{}, {}, {}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
    out = bootstrapped_returns(gamma, D, last_V)
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V)))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V))), 
         0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V)), 
         0.4 + gamma*(0.5 + gamma*last_V), 
         0.5 + gamma*last_V]
    assert np.allclose(out, y)
    
    D = Trajectory()
    dones = [False, False, False, True]
    infos = [{}, {}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [0.1, 0.2, 0.3, 0.4]
    out = bootstrapped_returns(gamma, D, last_V)
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*last_V*0.0))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*last_V*0.0)), 
         0.3 + gamma*(0.4 + gamma*last_V*0.0), 
         0.4 + gamma*last_V*0.0]
    assert np.allclose(out, y)
    
    D.step_infos[-1].done = False
    out = bootstrapped_returns(gamma, D, last_V)
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*last_V))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*last_V)), 
         0.3 + gamma*(0.4 + gamma*last_V), 
         0.4 + gamma*last_V]
    assert np.allclose(out, y)
    
    D = Trajectory()
    dones = [False, False, False, False, True]
    infos = [{}, {}, {}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
    out = bootstrapped_returns(gamma, D, last_V)
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V*0.0)))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V*0.0))), 
         0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V*0.0)), 
         0.4 + gamma*(0.5 + gamma*last_V*0.0),
         0.5 + gamma*last_V*0.0]
    assert np.allclose(out, y)
    
    D.step_infos[-1].done = False
    out = bootstrapped_returns(gamma, D, last_V)
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V)))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V))), 
         0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V)), 
         0.4 + gamma*(0.5 + gamma*last_V),
         0.5 + gamma*last_V]
    assert np.allclose(out, y)
    
    D = Trajectory()
    dones = [False, False]
    infos = [{}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [0.1, 0.2]
    out = bootstrapped_returns(gamma, D, last_V)
    y = [0.1 + gamma*(0.2 + gamma*last_V), 
         0.2 + gamma*last_V]
    assert np.allclose(out, y)
    

@pytest.mark.parametrize('gamma', [0.1, 0.5])
def test_td0_target(gamma):
    D = Trajectory()
    dones = [False, False, False, True]
    infos = [{}, {}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [0.1, 0.2, 0.3, 0.4]
    Vs = [1, 2, 3, 4]
    out = td0_target(gamma, D, Vs, 40)
    y = [0.1 + gamma*2, 
         0.2 + gamma*3,
         0.3 + gamma*4, 
         0.4 + gamma*40*0.0]
    assert np.allclose(out, y)
    
    D.step_infos[-1].done = False
    out = td0_target(gamma, D, Vs, 40)
    y = [0.1 + gamma*2, 
         0.2 + gamma*3,
         0.3 + gamma*4, 
         0.4 + gamma*40]
    assert np.allclose(out, y)

    D = Trajectory()
    dones = [False, False, False, False]
    infos = [{}, {}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [0.1, 0.2, 0.3, 0.4]
    Vs = [1, 2, 3, 4]
    out = td0_target(gamma, D, Vs, 40)
    y = [0.1 + gamma*2, 
         0.2 + gamma*3, 
         0.3 + gamma*4,
         0.4 + gamma*40]
    assert np.allclose(out, y)
    
    D = Trajectory()
    dones = [False, False, False, False, False, True]
    infos = [{}, {}, {}, {}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    Vs = [1, 2, 3, 4, 5, 6]
    out = td0_target(gamma, D, Vs, 60)
    y = [0.1 + gamma*2, 
         0.2 + gamma*3, 
         0.3 + gamma*4, 
         0.4 + gamma*5, 
         0.5 + gamma*6, 
         0.6 + gamma*60*0.0]
    assert np.allclose(out, y)
    
    D.step_infos[-1].done = False
    out = td0_target(gamma, D, Vs, 60)
    y = [0.1 + gamma*2, 
         0.2 + gamma*3, 
         0.3 + gamma*4, 
         0.4 + gamma*5, 
         0.5 + gamma*6, 
         0.6 + gamma*60]
    assert np.allclose(out, y)


@pytest.mark.parametrize('gamma', [0.1, 0.5])
def test_td0_error(gamma):
    D = Trajectory()
    dones = [False, False, False, True]
    infos = [{}, {}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [0.1, 0.2, 0.3, 0.4]
    Vs = [1, 2, 3, 4]
    out = td0_error(gamma, D, Vs, 40)
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2,
         0.3 + gamma*4 - 3, 
         0.4 + gamma*40*0.0 - 4]
    assert np.allclose(out, y)
    
    D.step_infos[-1].done = False
    out = td0_error(gamma, D, Vs, 40)
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2,
         0.3 + gamma*4 - 3, 
         0.4 + gamma*40 - 4]
    assert np.allclose(out, y)

    D = Trajectory()
    dones = [False, False, False, True]
    infos = [{}, {}, {}, {'TimeLimit.truncated': True}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [0.1, 0.2, 0.3, 0.4]
    Vs = [1, 2, 3, 4]
    out = td0_error(gamma, D, Vs, 40)
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2, 
         0.3 + gamma*4 - 3,
         0.4 + gamma*40 - 4]
    assert np.allclose(out, y)
    
    D = Trajectory()
    dones = [False, False, False, False, False, True]
    infos = [{}, {}, {}, {}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    Vs = [1, 2, 3, 4, 5, 6]
    out = td0_error(gamma, D, Vs, 60)
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2, 
         0.3 + gamma*4 - 3, 
         0.4 + gamma*5 - 4, 
         0.5 + gamma*6 - 5,  
         0.6 + gamma*60*0.0 - 6]
    assert np.allclose(out, y)
    
    D.step_infos[-1].done = False
    out = td0_error(gamma, D, Vs, 60)
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2, 
         0.3 + gamma*4 - 3, 
         0.4 + gamma*5 - 4, 
         0.5 + gamma*6 - 5,  
         0.6 + gamma*60 - 6]
    assert np.allclose(out, y)


def test_gae():
    D = Trajectory()
    dones = [False, False, True]
    infos = [{}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [1, 2, 3]
    Vs = [0.1, 1.1, 2.1]
    out = gae(1.0, 0.5, D, Vs, 10)
    assert np.allclose(out, [3.725, 3.45, 0.9])
    out = gae(0.1, 0.2, D, Vs, 10)
    assert np.allclose(out, [1.03256, 1.128, 0.9])
    
    D = Trajectory()
    dones = [False, False, True]
    infos = [{}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [1, 2, 3]
    Vs = [0.5, 1.5, 2.5]
    out = gae(1.0, 0.5, D, Vs, 99)
    assert np.allclose(out, [3.625, 3.25, 0.5])
    out = gae(0.1, 0.2, D, Vs, 99)
    assert np.allclose(out, [0.6652, 0.76, 0.5])

    D = Trajectory()
    dones = [False, False, False, False, False]
    infos = [{}, {}, {}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [1, 2, 3, 4, 5]
    Vs = [0.5, 1.5, 2.5, 3.5, 4.5]
    out = gae(1.0, 0.5, D, Vs, 20)
    assert np.allclose(out, [6.40625, 8.8125, 11.625, 15.25, 20.5])
    out = gae(0.1, 0.2, D, Vs, 20)
    assert np.allclose(out, [0.665348, 0.7674, 0.87, 1, 2.5])
    
    D = Trajectory()
    dones = [False, False, False, False, False]
    infos = [{}, {}, {}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [1, 2, 3, 4, 5]
    Vs = [0.1, 1.1, 2.1, 3.1, 4.1]
    out = gae(1.0, 0.5, D, Vs, 10)
    assert np.allclose(out, [5.80625, 7.6125, 9.225, 10.45, 10.9])
    out = gae(0.1, 0.2, D, Vs, 10)
    assert np.allclose(out, [1.03269478, 1.1347393, 1.23696, 1.348, 1.9])
    
    D = Trajectory()
    dones = [False, False, False, False, False, False, False, True]
    infos = [{}, {}, {}, {}, {}, {}, {}, {}]
    D.step_infos = [StepInfo(done, info) for done, info in zip(dones, infos)]
    D.rewards = [1, 2, 3, 4, 5, 6, 7, 8]
    Vs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    out = gae(1.0, 0.5, D, Vs, 30)
    assert np.allclose(out, [5.84375, 7.6875, 9.375, 10.75, 11.5, 11., 8, 0.])
    out = gae(0.1, 0.2, D, Vs, 30)
    assert np.allclose(out, [0.206164098, 0.308204915, 0.410245728, 0.5122864, 0.61432, 0.716, 0.8, 0])
