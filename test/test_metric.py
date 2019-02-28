import numpy as np

import torch

import pytest

from lagom.utils import Seeder

from lagom.envs import make_gym_env
from lagom.envs import make_vec_env
from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.vec_env import ParallelVecEnv

from lagom.envs import EnvSpec

from lagom.history import BatchEpisode
from lagom.history import BatchSegment

from lagom.history.metrics import final_state_from_episode
from lagom.history.metrics import final_state_from_segment
from lagom.history.metrics import terminal_state_from_episode
from lagom.history.metrics import terminal_state_from_segment
from lagom.history.metrics import returns_from_episode
from lagom.history.metrics import returns_from_segment
from lagom.history.metrics import bootstrapped_returns_from_episode
from lagom.history.metrics import bootstrapped_returns_from_segment
from lagom.history.metrics import td0_target_from_episode
from lagom.history.metrics import td0_target_from_segment
from lagom.history.metrics import td0_error_from_episode
from lagom.history.metrics import td0_error_from_segment
from lagom.history.metrics import gae_from_episode
from lagom.history.metrics import gae_from_segment


@pytest.mark.parametrize('num_env', [1, 3])
@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('mode', ['serial', 'parallel'])
@pytest.mark.parametrize('T', [1, 4, 6, 20])
@pytest.mark.parametrize('runner_class', [EpisodeRunner, RollingSegmentRunner])
def test_get_returns(num_env, init_seed, mode, T, runner_class)
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


@pytest.mark.parametrize('mode', ['serial', 'parallel'])
@pytest.mark.parametrize('gamma', [0.1, 0.5])
def test_get_bootstrapped_returns(mode, gamma):
    np.random.seed(3)

    make_env = lambda: SanityEnv()
    env = make_vec_env(make_env, 3, 0, mode)
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









        
    
   
    
def test_td0_target_from_episode():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)

    D = BatchEpisode(env_spec)
    D.r[0] = [1, 2, 3]
    D.done[0] = [False, False, True]
    D.completed[0] = True

    D.r[1] = [1, 2, 3, 4, 5]
    D.done[1] = [False, False, False, False, False]
    D.completed[1] = False

    D.r[2] = [1, 2, 3, 4, 5, 6, 7, 8]
    D.done[2] = [False, False, False, False, False, False, False, True]
    D.completed[2] = True

    all_Vs = [torch.tensor([[0.1], [0.5], [1.0]]), torch.tensor([[1.1], [1.5], [2.0]]), 
              torch.tensor([[2.1], [2.5], [3.0]]), torch.tensor([[3.1], [3.5], [4.0]]), 
              torch.tensor([[4.1], [4.5], [5.0]]), torch.tensor([[5.1], [5.5], [6.0]]),
              torch.tensor([[6.1], [6.5], [7.0]]), torch.tensor([[7.1], [7.5], [8.0]])]
    last_Vs = torch.tensor([10, 20, 30]).unsqueeze(1)

    all_Vs = torch.stack(all_Vs, 1)
    
    out = td0_target_from_episode(D, all_Vs, last_Vs, 1.0)
    assert out.shape == (3, D.maxT)
    assert np.allclose(out[0], [2.1, 4.1, 3, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [2.5, 4.5, 6.5, 8.5, 25, 0, 0, 0])
    assert np.allclose(out[2], [3, 5, 7, 9, 11, 13, 15, 8])
    del out
    
    out = td0_target_from_episode(D, all_Vs, last_Vs, 0.1)
    assert out.shape == (3, D.maxT)
    assert np.allclose(out[0], [1.11, 2.21, 3, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [1.15, 2.25, 3.35, 4.45, 7, 0, 0, 0])
    assert np.allclose(out[2], [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8])
    
    with pytest.raises(AssertionError):
        td0_target_from_segment(D, all_Vs, last_Vs, 0.1)
    
    
def test_td0_target_from_segment():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)

    D = BatchSegment(env_spec, 5)
    D.r[0] = [1, 2, 3, 4, 5]
    D.done[0] = [False, False, False, False, False]
    D.r[1] = [1, 2, 3, 4, 5]
    D.done[1] = [False, False, True, False, False]
    D.r[2] = [1, 2, 3, 4, 5]
    D.done[2] = [True, False, False, False, True]

    all_Vs = [torch.tensor([[0.1], [0.5], [1.0]]), torch.tensor([[1.1], [1.5], [2.0]]), 
              torch.tensor([[2.1], [2.5], [3.0]]), torch.tensor([[3.1], [3.5], [4.0]]), 
              torch.tensor([[4.1], [4.5], [5.0]])]
    last_Vs = torch.tensor([10, 20, 30]).unsqueeze(1)
    
    all_Vs = torch.stack(all_Vs, 1)
    
    out = td0_target_from_segment(D, all_Vs, last_Vs, 1.0)
    assert out.shape == (3, 5)
    assert np.allclose(out[0], [2.1, 4.1, 6.1, 8.1, 15])
    assert np.allclose(out[1], [2.5, 4.5, 3, 8.5, 25])
    assert np.allclose(out[2], [1, 5, 7, 9, 5])
    del out
    
    out = td0_target_from_segment(D, all_Vs, last_Vs, 0.1)
    assert out.shape == (3, 5)
    assert np.allclose(out[0], [1.11, 2.21, 3.31, 4.41, 6])
    assert np.allclose(out[1], [1.15, 2.25, 3, 4.45, 7])
    assert np.allclose(out[2], [1, 2.3, 3.4, 4.5, 5])
    
    with pytest.raises(AssertionError):
        td0_target_from_episode(D, all_Vs, last_Vs, 0.1)
        
        
def test_td0_error_from_episode():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)

    D = BatchEpisode(env_spec)
    D.r[0] = [1, 2, 3]
    D.done[0] = [False, False, True]
    D.completed[0] = True

    D.r[1] = [1, 2, 3, 4, 5]
    D.done[1] = [False, False, False, False, False]
    D.completed[1] = False

    D.r[2] = [1, 2, 3, 4, 5, 6, 7, 8]
    D.done[2] = [False, False, False, False, False, False, False, True]
    D.completed[2] = True

    all_Vs = [torch.tensor([[0.1], [0.5], [1.0]]), torch.tensor([[1.1], [1.5], [2.0]]), 
              torch.tensor([[2.1], [2.5], [3.0]]), torch.tensor([[3.1], [3.5], [4.0]]), 
              torch.tensor([[4.1], [4.5], [5.0]]), torch.tensor([[5.1], [5.5], [6.0]]),
              torch.tensor([[6.1], [6.5], [7.0]]), torch.tensor([[7.1], [7.5], [8.0]])]
    last_Vs = torch.tensor([10, 20, 30]).unsqueeze(1)
    
    all_Vs = torch.stack(all_Vs, 1)
    
    out = td0_error_from_episode(D, all_Vs, last_Vs, 1.0)
    assert out.shape == (3, D.maxT)
    assert np.allclose(out[0], [2.0, 3, 0.9, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [2, 3, 4, 5, 20.5, 0, 0, 0])
    assert np.allclose(out[2], [2, 3, 4, 5, 6, 7, 8, 0])
    del out
    
    out = td0_error_from_episode(D, all_Vs, last_Vs, 0.1)
    assert out.shape == (3, D.maxT)
    assert np.allclose(out[0], [1.01, 1.11, 0.9, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [0.65, 0.75, 0.85, 0.95, 2.5, 0, 0, 0])
    assert np.allclose(out[2], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0])
    
    with pytest.raises(AssertionError):
        td0_error_from_segment(D, all_Vs, last_Vs, 0.1)
        
        
def test_td0_error_from_segment():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)

    D = BatchSegment(env_spec, 5)
    D.r[0] = [1, 2, 3, 4, 5]
    D.done[0] = [False, False, False, False, False]
    D.r[1] = [1, 2, 3, 4, 5]
    D.done[1] = [False, False, True, False, False]
    D.r[2] = [1, 2, 3, 4, 5]
    D.done[2] = [True, False, False, False, True]

    all_Vs = [torch.tensor([[0.1], [0.5], [1.0]]), torch.tensor([[1.1], [1.5], [2.0]]), 
              torch.tensor([[2.1], [2.5], [3.0]]), torch.tensor([[3.1], [3.5], [4.0]]), 
              torch.tensor([[4.1], [4.5], [5.0]])]
    last_Vs = torch.tensor([10, 20, 30]).unsqueeze(1)
    
    all_Vs = torch.stack(all_Vs, 1)
    
    out = td0_error_from_segment(D, all_Vs, last_Vs, 1.0)
    assert out.shape == (3, 5)
    assert np.allclose(out[0], [2.0, 3, 4, 5, 10.9])
    assert np.allclose(out[1], [2, 3, 0.5, 5, 20.5])
    assert np.allclose(out[2], [0, 3, 4, 5, 0])
    del out
    
    out = td0_error_from_segment(D, all_Vs, last_Vs, 0.1)
    assert out.shape == (3, 5)
    assert np.allclose(out[0], [1.01, 1.11, 1.21, 1.31, 1.9])
    assert np.allclose(out[1], [0.65, 0.75, 0.5, 0.95, 2.5])
    assert np.allclose(out[2], [0, 0.3, 0.4, 0.5, 0])
    
    with pytest.raises(AssertionError):
        td0_error_from_episode(D, all_Vs, last_Vs, 0.1)
    
    
def test_gae_from_episode():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)

    D = BatchEpisode(env_spec)
    D.r[0] = [1, 2, 3]
    D.done[0] = [False, False, True]
    D.completed[0] = True

    D.r[1] = [1, 2, 3, 4, 5]
    D.done[1] = [False, False, False, False, False]
    D.completed[1] = False

    D.r[2] = [1, 2, 3, 4, 5, 6, 7, 8]
    D.done[2] = [False, False, False, False, False, False, False, True]
    D.completed[2] = True

    all_Vs = [torch.tensor([[0.1], [0.5], [1.0]]), torch.tensor([[1.1], [1.5], [2.0]]), 
              torch.tensor([[2.1], [2.5], [3.0]]), torch.tensor([[3.1], [3.5], [4.0]]), 
              torch.tensor([[4.1], [4.5], [5.0]]), torch.tensor([[5.1], [5.5], [6.0]]),
              torch.tensor([[6.1], [6.5], [7.0]]), torch.tensor([[7.1], [7.5], [8.0]])]
    last_Vs = torch.tensor([10, 20, 30]).unsqueeze(1)
    
    all_Vs = torch.stack(all_Vs, 1)
    
    out = gae_from_episode(D, all_Vs, last_Vs, 1.0, 0.5)
    assert out.shape == (3, D.maxT)
    assert np.allclose(out[0], [3.725, 3.45, 0.9, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [6.40625, 8.8125, 11.625, 15.25, 20.5, 0, 0, 0])
    assert np.allclose(out[2], [5.84375, 7.6875, 9.375, 10.75, 11.5, 11., 8, 0.])
    del out
    
    out = gae_from_episode(D, all_Vs, last_Vs, 0.1, 0.2)
    assert out.shape == (3, D.maxT)
    assert np.allclose(out[0], [1.03256, 1.128, 0.9, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [0.665348, 0.7674, 0.87, 1, 2.5, 0, 0, 0])
    assert np.allclose(out[2], [0.206164098, 0.308204915, 0.410245728, 0.5122864, 0.61432, 0.716, 0.8, 0])
    
    with pytest.raises(AssertionError):
        gae_from_segment(D, all_Vs, last_Vs, 0.1, 0.2)
        

def test_gae_from_segment():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)

    D = BatchSegment(env_spec, 5)
    D.r[0] = [1, 2, 3, 4, 5]
    D.done[0] = [False, False, False, False, False]
    D.r[1] = [1, 2, 3, 4, 5]
    D.done[1] = [False, False, True, False, False]
    D.r[2] = [1, 2, 3, 4, 5]
    D.done[2] = [True, False, False, False, True]

    all_Vs = [torch.tensor([[0.1], [0.5], [1.0]]), torch.tensor([[1.1], [1.5], [2.0]]), 
              torch.tensor([[2.1], [2.5], [3.0]]), torch.tensor([[3.1], [3.5], [4.0]]), 
              torch.tensor([[4.1], [4.5], [5.0]])]
    last_Vs = torch.tensor([10, 20, 30]).unsqueeze(1)
    
    all_Vs = torch.stack(all_Vs, 1)
    
    out = gae_from_segment(D, all_Vs, last_Vs, 1.0, 0.5)
    assert out.shape == (3, 5)
    assert np.allclose(out[0], [5.80625, 7.6125, 9.225, 10.45, 10.9])
    assert np.allclose(out[1], [3.625, 3.25, 0.5, 15.25, 20.5])
    assert np.allclose(out[2], [0, 6.25, 6.5, 5, 0])
    del out
    
    out = gae_from_segment(D, all_Vs, last_Vs, 0.1, 0.2)
    assert out.shape == (3, 5)
    assert np.allclose(out[0], [1.03269478, 1.1347393, 1.23696, 1.348, 1.9])
    assert np.allclose(out[1], [0.6652, 0.76, 0.5, 1, 2.5])
    assert np.allclose(out[2], [0, 0.3082, 0.41, 0.5, 0])
    
    with pytest.raises(AssertionError):
        gae_from_episode(D, all_Vs, last_Vs, 0.1, 0.2)
