import pytest
import numpy as np
import torch

import gym
from gym.wrappers import TimeLimit

from lagom import RandomAgent
from lagom import EpisodeRunner
from lagom import StepRunner
from lagom.utils import numpify

from lagom.metric import returns
from lagom.metric import bootstrapped_returns
from lagom.metric import td0_target
from lagom.metric import td0_error
from lagom.metric import gae
from lagom.metric import vtrace

from .sanity_env import SanityEnv


@pytest.mark.parametrize('gamma', [0.1, 0.99, 1.0])
def test_returns(gamma):
    assert np.allclose(returns(1.0, [1, 2, 3]), [6, 5, 3])
    assert np.allclose(returns(0.1, [1, 2, 3]), [1.23, 2.3, 3])
    assert np.allclose(returns(1.0, [1, 2, 3, 4, 5]), [15, 14, 12, 9, 5])
    assert np.allclose(returns(0.1, [1, 2, 3, 4, 5]), [1.2345, 2.345, 3.45, 4.5, 5])
    assert np.allclose(returns(1.0, [1, 2, 3, 4, 5, 6, 7, 8]), [36, 35, 33, 30, 26, 21, 15, 8])
    assert np.allclose(returns(0.1, [1, 2, 3, 4, 5, 6, 7, 8]), [1.2345678, 2.345678, 3.45678, 4.5678, 5.678, 6.78, 7.8, 8])
    
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
    assert np.allclose(returns(gamma, [0.1]), y1)
    assert np.allclose(returns(gamma, [0.1, 0.2]), y2)
    assert np.allclose(returns(gamma, [0.1, 0.2, 0.3]), y3)
    assert np.allclose(returns(gamma, [0.1, 0.2, 0.3, 0.4]), y4)
    assert np.allclose(returns(gamma, [0.1, 0.2, 0.3, 0.4, 0.5]), y5)
    assert np.allclose(returns(gamma, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), y6)


@pytest.mark.parametrize('gamma', [0.1, 0.99, 1.0])
@pytest.mark.parametrize('last_V', [-3.0, 0.0, 2.0])
def test_bootstrapped_returns(gamma, last_V):
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*last_V))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*last_V)), 
         0.3 + gamma*(0.4 + gamma*last_V), 
         0.4 + gamma*last_V]
    reach_terminal = False
    rewards = [0.1, 0.2, 0.3, 0.4]
    assert np.allclose(bootstrapped_returns(gamma, rewards, last_V, reach_terminal), y)
    assert np.allclose(bootstrapped_returns(gamma, rewards, torch.tensor(last_V), reach_terminal), y)
    
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*last_V*0.0))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*last_V*0.0)), 
         0.3 + gamma*(0.4 + gamma*last_V*0.0), 
         0.4 + gamma*last_V*0.0]
    reach_terminal = True
    rewards = [0.1, 0.2, 0.3, 0.4]
    assert np.allclose(bootstrapped_returns(gamma, rewards, last_V, reach_terminal), y)
    assert np.allclose(bootstrapped_returns(gamma, rewards, torch.tensor(last_V), reach_terminal), y)
    
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V)))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V))), 
         0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V)), 
         0.4 + gamma*(0.5 + gamma*last_V), 
         0.5 + gamma*last_V]
    reach_terminal = False
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert np.allclose(bootstrapped_returns(gamma, rewards, last_V, reach_terminal), y)
    assert np.allclose(bootstrapped_returns(gamma, rewards, torch.tensor(last_V), reach_terminal), y)
    
    y = [0.1 + gamma*(0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V*0.0)))), 
         0.2 + gamma*(0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V*0.0))), 
         0.3 + gamma*(0.4 + gamma*(0.5 + gamma*last_V*0.0)), 
         0.4 + gamma*(0.5 + gamma*last_V*0.0),
         0.5 + gamma*last_V*0.0]
    reach_terminal = True
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert np.allclose(bootstrapped_returns(gamma, rewards, last_V, reach_terminal), y)
    assert np.allclose(bootstrapped_returns(gamma, rewards, torch.tensor(last_V), reach_terminal), y)


@pytest.mark.parametrize('gamma', [0.1, 0.99, 1.0])
@pytest.mark.parametrize('last_V', [-3.0, 0.0, 2.0])
def test_td0_target(gamma, last_V):
    y = [0.1 + gamma*2, 
         0.2 + gamma*3,
         0.3 + gamma*4, 
         0.4 + gamma*last_V*0.0]
    rewards = [0.1, 0.2, 0.3, 0.4]
    Vs = [1, 2, 3, 4]
    reach_terminal = True
    assert np.allclose(td0_target(gamma, rewards, Vs, last_V, reach_terminal), y)
    assert np.allclose(td0_target(gamma, rewards, torch.tensor(Vs), torch.tensor(last_V), reach_terminal), y)
    
    y = [0.1 + gamma*2, 
         0.2 + gamma*3,
         0.3 + gamma*4, 
         0.4 + gamma*last_V]
    rewards = [0.1, 0.2, 0.3, 0.4]
    Vs = [1, 2, 3, 4]
    reach_terminal = False
    assert np.allclose(td0_target(gamma, rewards, Vs, last_V, reach_terminal), y)
    assert np.allclose(td0_target(gamma, rewards, torch.tensor(Vs), torch.tensor(last_V), reach_terminal), y)
    
    y = [0.1 + gamma*2, 
         0.2 + gamma*3, 
         0.3 + gamma*4, 
         0.4 + gamma*5, 
         0.5 + gamma*6, 
         0.6 + gamma*last_V*0.0]
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    Vs = [1, 2, 3, 4, 5, 6]
    reach_terminal = True
    assert np.allclose(td0_target(gamma, rewards, Vs, last_V, reach_terminal), y)
    assert np.allclose(td0_target(gamma, rewards, torch.tensor(Vs), torch.tensor(last_V), reach_terminal), y)
    
    y = [0.1 + gamma*2, 
         0.2 + gamma*3, 
         0.3 + gamma*4, 
         0.4 + gamma*5, 
         0.5 + gamma*6, 
         0.6 + gamma*last_V]
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    Vs = [1, 2, 3, 4, 5, 6]
    reach_terminal = False
    assert np.allclose(td0_target(gamma, rewards, Vs, last_V, reach_terminal), y)
    assert np.allclose(td0_target(gamma, rewards, torch.tensor(Vs), torch.tensor(last_V), reach_terminal), y)


@pytest.mark.parametrize('gamma', [0.1, 0.99, 1.0])
@pytest.mark.parametrize('last_V', [-3.0, 0.0, 2.0])
def test_td0_error(gamma, last_V):
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2,
         0.3 + gamma*4 - 3, 
         0.4 + gamma*last_V*0.0 - 4]
    rewards = [0.1, 0.2, 0.3, 0.4]
    Vs = [1, 2, 3, 4]
    reach_terminal = True
    assert np.allclose(td0_error(gamma, rewards, Vs, last_V, reach_terminal), y)
    assert np.allclose(td0_error(gamma, rewards, torch.tensor(Vs), torch.tensor(last_V), reach_terminal), y)
    
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2,
         0.3 + gamma*4 - 3, 
         0.4 + gamma*last_V - 4]
    rewards = [0.1, 0.2, 0.3, 0.4]
    Vs = [1, 2, 3, 4]
    reach_terminal = False
    assert np.allclose(td0_error(gamma, rewards, Vs, last_V, reach_terminal), y)
    assert np.allclose(td0_error(gamma, rewards, torch.tensor(Vs), torch.tensor(last_V), reach_terminal), y)
    
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2, 
         0.3 + gamma*4 - 3, 
         0.4 + gamma*5 - 4, 
         0.5 + gamma*6 - 5,  
         0.6 + gamma*last_V*0.0 - 6]
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    Vs = [1, 2, 3, 4, 5, 6]
    reach_terminal = True
    assert np.allclose(td0_error(gamma, rewards, Vs, last_V, reach_terminal), y)
    assert np.allclose(td0_error(gamma, rewards, torch.tensor(Vs), torch.tensor(last_V), reach_terminal), y)
    
    y = [0.1 + gamma*2 - 1, 
         0.2 + gamma*3 - 2, 
         0.3 + gamma*4 - 3, 
         0.4 + gamma*5 - 4, 
         0.5 + gamma*6 - 5,  
         0.6 + gamma*last_V - 6]
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    Vs = [1, 2, 3, 4, 5, 6]
    reach_terminal = False
    assert np.allclose(td0_error(gamma, rewards, Vs, last_V, reach_terminal), y)
    assert np.allclose(td0_error(gamma, rewards, torch.tensor(Vs), torch.tensor(last_V), reach_terminal), y)


def test_gae():
    rewards = [1, 2, 3]
    Vs = [0.1, 1.1, 2.1]
    assert np.allclose(gae(1.0, 0.5, rewards, Vs, 10, True), 
                       [3.725, 3.45, 0.9])
    assert np.allclose(gae(1.0, 0.5, rewards, torch.tensor(Vs), torch.tensor(10), True), 
                       [3.725, 3.45, 0.9])
    assert np.allclose(gae(0.1, 0.2, rewards, Vs, 10, True), 
                       [1.03256, 1.128, 0.9])
    assert np.allclose(gae(0.1, 0.2, rewards, torch.tensor(Vs), torch.tensor(10), True), 
                       [1.03256, 1.128, 0.9])
    
    rewards = [1, 2, 3]
    Vs = [0.5, 1.5, 2.5]
    assert np.allclose(gae(1.0, 0.5, rewards, Vs, 99, True), 
                       [3.625, 3.25, 0.5])
    assert np.allclose(gae(1.0, 0.5, rewards, torch.tensor(Vs), torch.tensor(99), True), 
                       [3.625, 3.25, 0.5])
    assert np.allclose(gae(0.1, 0.2, rewards, Vs, 99, True), 
                       [0.6652, 0.76, 0.5])
    assert np.allclose(gae(0.1, 0.2, rewards, torch.tensor(Vs), torch.tensor(99), True), 
                       [0.6652, 0.76, 0.5])
    
    rewards = [1, 2, 3, 4, 5]
    Vs = [0.5, 1.5, 2.5, 3.5, 4.5]
    assert np.allclose(gae(1.0, 0.5, rewards, Vs, 20, False), 
                       [6.40625, 8.8125, 11.625, 15.25, 20.5])
    assert np.allclose(gae(1.0, 0.5, rewards, torch.tensor(Vs), torch.tensor(20), False), 
                       [6.40625, 8.8125, 11.625, 15.25, 20.5])
    assert np.allclose(gae(0.1, 0.2, rewards, Vs, 20, False), 
                       [0.665348, 0.7674, 0.87, 1, 2.5])
    assert np.allclose(gae(0.1, 0.2, rewards, torch.tensor(Vs), torch.tensor(20), False), 
                       [0.665348, 0.7674, 0.87, 1, 2.5])

    rewards = [1, 2, 3, 4, 5]
    Vs = [0.1, 1.1, 2.1, 3.1, 4.1]
    assert np.allclose(gae(1.0, 0.5, rewards, Vs, 10, False), 
                       [5.80625, 7.6125, 9.225, 10.45, 10.9])
    assert np.allclose(gae(1.0, 0.5, rewards, torch.tensor(Vs), torch.tensor(10), False), 
                       [5.80625, 7.6125, 9.225, 10.45, 10.9])
    assert np.allclose(gae(0.1, 0.2, rewards, Vs, 10, False), 
                       [1.03269478, 1.1347393, 1.23696, 1.348, 1.9])
    assert np.allclose(gae(0.1, 0.2, rewards, torch.tensor(Vs), torch.tensor(10), False), 
                       [1.03269478, 1.1347393, 1.23696, 1.348, 1.9])
    
    rewards = [1, 2, 3, 4, 5, 6, 7, 8]
    Vs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert np.allclose(gae(1.0, 0.5, rewards, Vs, 30, True), 
                       [5.84375, 7.6875, 9.375, 10.75, 11.5, 11., 8, 0.])
    assert np.allclose(gae(1.0, 0.5, rewards, torch.tensor(Vs), torch.tensor(30), True), 
                       [5.84375, 7.6875, 9.375, 10.75, 11.5, 11., 8, 0.])
    assert np.allclose(gae(0.1, 0.2, rewards, Vs, 30, True), 
                       [0.206164098, 0.308204915, 0.410245728, 0.5122864, 0.61432, 0.716, 0.8, 0])
    assert np.allclose(gae(0.1, 0.2, rewards, torch.tensor(Vs), torch.tensor(30), True), 
                       [0.206164098, 0.308204915, 0.410245728, 0.5122864, 0.61432, 0.716, 0.8, 0])


@pytest.mark.parametrize('gamma', [0.1, 1.0])
@pytest.mark.parametrize('last_V', [0.3, [0.5]])
@pytest.mark.parametrize('reach_terminal', [True, False])
@pytest.mark.parametrize('clip_rho', [0.5, 1.0])
@pytest.mark.parametrize('clip_pg_rho', [0.3, 1.1])
def test_vtrace(gamma, last_V, reach_terminal, clip_rho, clip_pg_rho):
    behavior_logprobs = [1, 2, 3]
    target_logprobs = [4, 5, 6]
    Rs = [7, 8, 9]
    Vs = [10, 11, 12]
    
    vs_test, As_test = vtrace(behavior_logprobs, target_logprobs, gamma, Rs, Vs, last_V, reach_terminal, clip_rho, clip_pg_rho)
    
    # ground truth calculation
    behavior_logprobs = numpify(behavior_logprobs, np.float32)
    target_logprobs = numpify(target_logprobs, np.float32)
    Rs = numpify(Rs, np.float32)
    Vs = numpify(Vs, np.float32)
    last_V = numpify(last_V, np.float32)
    
    rhos = np.exp(target_logprobs - behavior_logprobs)
    clipped_rhos = np.minimum(clip_rho, rhos)
    cs = np.minimum(1.0, rhos)
    deltas = clipped_rhos*td0_error(gamma, Rs, Vs, last_V, reach_terminal)
    
    vs = np.array([Vs[0] + gamma**0*1*deltas[0] + gamma*cs[0]*deltas[1] + gamma**2*cs[0]*cs[1]*deltas[2], 
                   Vs[1] + gamma**0*1*deltas[1] + gamma*cs[1]*deltas[2], 
                   Vs[2] + gamma**0*1*deltas[2]])
    vs_next = np.append(vs[1:], (1. - reach_terminal)*last_V)
    clipped_pg_rhos = np.minimum(clip_pg_rho, rhos)
    As = clipped_pg_rhos*(Rs + gamma*vs_next - Vs)
    
    assert np.allclose(vs, vs_test)
    assert np.allclose(As, As_test)
