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


@pytest.mark.parametrize('vec_env', [SerialVecEnv, ParallelVecEnv])
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_batch_episode(vec_env, env_id):
    env = make_vec_env(vec_env, make_gym_env, env_id, 3, 0)
    env_spec = EnvSpec(env)

    D = BatchEpisode(env_spec)

    if env_id == 'CartPole-v1':
        sticky_action = 1
        action_shape = ()
        action_dtype = np.int32
    elif env_id == 'Pendulum-v0':
        sticky_action = [0.1]
        action_shape = env_spec.action_space.shape
        action_dtype = np.float32

    obs = env.reset()
    D.add_observation(obs)
    for t in range(30):
        action = [sticky_action]*env.num_env
        obs, reward, done, info = env.step(action)
        D.add_observation(obs)
        D.add_action(action)
        D.add_reward(reward)
        D.add_done(done)
        D.add_info(info)
        D.add_batch_info({'V': [0.1*(t+1), (t+1), 10*(t+1)]})
        [D.set_completed(n) for n, d in enumerate(done) if d]

    assert D.N == 3
    assert len(D.Ts) == 3
    assert D.maxT == max(D.Ts)
    
    assert all([isinstance(x, np.ndarray) for x in [D.numpy_observations, D.numpy_actions, D.numpy_rewards, D.numpy_dones, D.numpy_masks]])
    assert all([x.dtype == np.float32 for x in [D.numpy_observations, D.numpy_rewards, D.numpy_masks]])
    assert all([x.shape == (3, D.maxT) for x in [D.numpy_rewards, D.numpy_dones, D.numpy_masks]])
    assert D.numpy_actions.dtype == action_dtype
    assert D.numpy_dones.dtype == np.bool
    assert D.numpy_observations.shape == (3, D.maxT+1) + env_spec.observation_space.shape
    assert D.numpy_actions.shape == (3, D.maxT) + action_shape
    assert isinstance(D.batch_infos, list) and len(D.batch_infos) == 30
    assert np.allclose([0.1*(x+1) for x in range(30)], [info['V'][0] for info in D.batch_infos])
    assert np.allclose([1*(x+1) for x in range(30)], [info['V'][1] for info in D.batch_infos])
    assert np.allclose([10*(x+1) for x in range(30)], [info['V'][2] for info in D.batch_infos])
    
    seeder = Seeder(0)
    seed1, seed2, seed3 = seeder(3)
    env1 = make_gym_env(env_id, seed1)
    env2 = make_gym_env(env_id, seed2)
    env3 = make_gym_env(env_id, seed3)

    for n, ev in enumerate([env1, env2, env3]):
        obs = ev.reset()
        assert np.allclose(obs, D.observations[n][0])
        assert np.allclose(obs, D.numpy_observations[n, 0, ...])
        for t in range(30):
            obs, reward, done, info = ev.step(sticky_action)

            assert np.allclose(reward, D.rewards[n][t])
            assert np.allclose(reward, D.numpy_rewards[n, t])
            assert np.allclose(done, D.dones[n][t])
            assert done == D.numpy_dones[n, t]
            assert int(not done) == D.masks[n][t]
            assert int(not done) == D.numpy_masks[n, t]

            if done:
                assert np.allclose(obs, D.infos[n][t]['terminal_observation'])
                assert D.completes[n]
                assert np.allclose(0.0, D.numpy_observations[n, t+1+1:, ...])
                assert np.allclose(0.0, D.numpy_actions[n, t+1:, ...])
                assert np.allclose(0.0, D.numpy_rewards[n, t+1:])
                assert np.allclose(True, D.numpy_dones[n, t+1:])
                assert np.allclose(0.0, D.numpy_masks[n, t+1:])
                break
            else:
                assert np.allclose(obs, D.observations[n][t+1])
                
                
@pytest.mark.parametrize('vec_env', [SerialVecEnv, ParallelVecEnv])
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_batch_segment(vec_env, env_id):
    env = make_vec_env(vec_env, make_gym_env, env_id, 3, 0)
    env_spec = EnvSpec(env)

    T = 30

    D = BatchSegment(env_spec, T)

    if env_id == 'CartPole-v1':
        sticky_action = 1
        action_shape = ()
        action_dtype = np.int32
    elif env_id == 'Pendulum-v0':
        sticky_action = [0.1]
        action_shape = env_spec.action_space.shape
        action_dtype = np.float32

    obs = env.reset()
    D.add_observation(0, obs)
    for t in range(T):
        action = [sticky_action]*env.num_env
        obs, reward, done, info = env.step(action)
        D.add_observation(t+1, obs)
        D.add_action(t, action)
        D.add_reward(t, reward)
        D.add_done(t, done)
        D.add_info(info)
        D.add_batch_info({'V': [0.1*(t+1), (t+1), 10*(t+1)]})

    assert D.N == 3
    assert D.T == T
    assert all([isinstance(x, np.ndarray) for x in [D.numpy_observations, D.numpy_actions, 
                                                    D.numpy_rewards, D.numpy_dones, D.numpy_masks]])
    assert all([x.dtype == np.float32 for x in [D.numpy_observations, D.numpy_rewards, D.numpy_masks]])
    assert D.numpy_actions.dtype == action_dtype
    assert D.numpy_dones.dtype == np.bool
    assert D.numpy_observations.shape[:2] == (3, T+1)
    assert D.numpy_actions.shape == (3, T) + action_shape
    assert all([x.shape == (3, T) for x in [D.numpy_rewards, D.numpy_dones, D.numpy_masks]])
    assert isinstance(D.batch_infos, list) and len(D.batch_infos) == T
    assert np.allclose([0.1*(x+1) for x in range(T)], [info['V'][0] for info in D.batch_infos])
    assert np.allclose([1*(x+1) for x in range(T)], [info['V'][1] for info in D.batch_infos])
    assert np.allclose([10*(x+1) for x in range(T)], [info['V'][2] for info in D.batch_infos])

    seeder = Seeder(0)
    seed1, seed2, seed3 = seeder(3)
    env1 = make_gym_env(env_id, seed1)
    env2 = make_gym_env(env_id, seed2)
    env3 = make_gym_env(env_id, seed3)

    for n, ev in enumerate([env1, env2, env3]):
        obs = ev.reset()
        assert np.allclose(obs, D.numpy_observations[n, 0, ...])
        for t in range(T):
            obs, reward, done, info = ev.step(sticky_action)
            if done:
                info['terminal_observation'] = obs
                obs = ev.reset()

            assert np.allclose(obs, D.numpy_observations[n, t+1, ...])
            assert np.allclose(sticky_action, D.numpy_actions[n, t, ...])
            assert np.allclose(reward, D.numpy_rewards[n, t])
            assert done == D.numpy_dones[n, t]
            assert int(not done) == D.numpy_masks[n, t]

            if done:
                assert np.allclose(info['terminal_observation'], D.infos[n][t]['terminal_observation'])


def test_final_state_from_episode():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)

    with pytest.raises(AssertionError):
        final_state_from_episode([1, 2, 3])

    D = BatchEpisode(env_spec)
    D.obs[0] = [0.1, 0.2, 1.3]
    D.done[0] = [False, False, True]
    D.info[0] = [{}, {}, {'terminal_observation': 0.3}]

    D.obs[1] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    D.done[1] = [False]*9

    D.obs[2] = [10, 15]
    D.done[2] = [False, True]
    D.info[2] = [{}, {'terminal_observation': 20}]

    final_states = final_state_from_episode(D)
    
    assert final_states.shape == (3,) + env_spec.observation_space.shape
    assert np.allclose(final_states[0], 0.3)
    assert np.allclose(final_states[1], 9)
    assert np.allclose(final_states[2], 20)
    
    with pytest.raises(AssertionError):
        final_state_from_segment(D)
                

def test_final_state_from_segment():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)

    with pytest.raises(AssertionError):
        final_state_from_segment([1, 2, 3])

    D = BatchSegment(env_spec, 4)
    D.obs = np.random.randn(*D.obs.shape)
    D.done.fill(False)

    D.done[0, -1] = True
    D.info[0] = [{}, {}, {}, {'terminal_observation': [0.1, 0.2, 0.3, 0.4]}]

    D.done[1, 2] = True
    D.info[1] = [{}, {}, {'terminal_observation': [1, 2, 3, 4]}, {}]

    D.done[2, -1] = True
    D.info[2] = [{}, {}, {}, {'terminal_observation': [10, 20, 30, 40]}]


    final_states = final_state_from_segment(D)
    assert final_states.shape == (3, ) + env_spec.observation_space.shape
    assert np.allclose(final_states[0], [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(final_states[1], D.numpy_observations[1, -1, ...])
    assert not np.allclose(final_states[1], [1, 2, 3, 4])
    assert np.allclose(final_states[2], [10, 20, 30, 40])
    
    with pytest.raises(AssertionError):
        final_state_from_episode(D)
    
    
def test_terminal_state_from_episode():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)

    with pytest.raises(AssertionError):
        terminal_state_from_episode([1, 2, 3])

    D = BatchEpisode(env_spec)
    D.obs[0] = [0.1, 0.2, 1.3]
    D.done[0] = [False, False, True]
    D.info[0] = [{}, {}, {'terminal_observation': 0.3}]

    D.obs[1] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    D.done[1] = [False]*9

    D.obs[2] = [10, 15]
    D.done[2] = [False, True]
    D.info[2] = [{}, {'terminal_observation': 20}]

    terminal_states = terminal_state_from_episode(D)
    assert terminal_states.shape == (2,) + env_spec.observation_space.shape
    assert np.allclose(terminal_states[0], 0.3)
    assert np.allclose(terminal_states[1], 20)

    D.done[0][-1] = False
    D.done[2][-1] = False
    assert terminal_state_from_episode(D) is None
    
    with pytest.raises(AssertionError):
        terminal_state_from_segment(D)
    
    
def test_terminal_state_from_segment():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)

    with pytest.raises(AssertionError):
        terminal_state_from_segment([1, 2, 3])

    D = BatchSegment(env_spec, 4)
    D.obs = np.random.randn(*D.obs.shape)
    D.done.fill(False)

    D.done[0, -1] = True
    D.info[0] = [{}, {}, {}, {'terminal_observation': [0.1, 0.2, 0.3, 0.4]}]

    D.done[1, 2] = True
    D.info[1] = [{}, {}, {'terminal_observation': [1, 2, 3, 4]}, {}]

    D.done[2, -1] = True
    D.info[2] = [{}, {}, {}, {'terminal_observation': [10, 20, 30, 40]}]

    terminal_states = terminal_state_from_segment(D)

    assert terminal_states.shape == (3, 4)
    assert np.allclose(terminal_states[0], [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(terminal_states[1], [1, 2, 3, 4])
    assert np.allclose(terminal_states[2], [10, 20, 30, 40])

    D.done[0, -1] = False
    D.done[1, 2] = False
    terminal_states = terminal_state_from_segment(D)
    assert terminal_states.shape == (1, 4)
    assert np.allclose(terminal_states[0], [10, 20, 30, 40])

    D.done[2, -1] = False
    assert terminal_state_from_segment(D) is None
    
    with pytest.raises(AssertionError):
        terminal_state_from_episode(D)


def test_returns_from_episode():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)
    
    D = BatchEpisode(env_spec)
    D.r[0] = [1, 2, 3]
    D.done[0] = [False, False, True]
    D.r[1] = [1, 2, 3, 4, 5]
    D.done[1] = [False, False, False, False, False]
    D.r[2] = [1, 2, 3, 4, 5, 6, 7, 8]
    D.done[2] = [False, False, False, False, False, False, False, True]

    out = returns_from_episode(D, 1.0)
    assert out.shape == (3, D.maxT)
    assert np.allclose(out[0], [6, 5, 3, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [15, 14, 12, 9, 5, 0, 0, 0])
    assert np.allclose(out[2], [36, 35, 33, 30, 26, 21, 15, 8])
    del out
    
    out = returns_from_episode(D, 0.1)
    assert out.shape == (3, D.maxT)
    assert np.allclose(out[0], [1.23, 2.3, 3, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [1.2345, 2.345, 3.45, 4.5, 5, 0, 0, 0])
    assert np.allclose(out[2], [1.2345678, 2.345678, 3.45678, 4.5678, 5.678, 6.78, 7.8, 8])
    
    with pytest.raises(AssertionError):
        returns_from_segment(D, 0.1)
    
    
def test_returns_from_segment():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)

    D = BatchSegment(env_spec, 5)
    D.r[0] = [1, 2, 3, 4, 5]
    D.done[0] = [False, False, False, False, False]
    D.r[1] = [1, 2, 3, 4, 5]
    D.done[1] = [False, False, True, False, False]
    D.r[2] = [1, 2, 3, 4, 5]
    D.done[2] = [True, False, False, False, True]

    out = returns_from_segment(D, 1.0)
    assert out.shape == (3, 5)
    assert np.allclose(out[0], [15, 14, 12, 9, 5])
    assert np.allclose(out[1], [6, 5, 3, 9, 5])
    assert np.allclose(out[2], [1, 14, 12, 9, 5])
    del out
    
    out = returns_from_segment(D, 0.1)
    assert out.shape == (3, 5)
    assert np.allclose(out[0], [1.2345, 2.345, 3.45, 4.5, 5])
    assert np.allclose(out[1], [1.23, 2.3, 3, 4.5, 5])
    assert np.allclose(out[2], [1, 2.345, 3.45, 4.5, 5])
    
    with pytest.raises(AssertionError):
        returns_from_episode(D, 0.1)
    
    
def test_bootstrapped_returns_from_episode():
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
    
    last_Vs = torch.tensor([10, 20, 30]).unsqueeze(1)
    
    out = bootstrapped_returns_from_episode(D, last_Vs, 1.0)
    assert out.shape == (3, D.maxT)
    assert np.allclose(out[0], [6, 5, 3, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [35, 34, 32, 29, 25, 0, 0, 0])
    assert np.allclose(out[2], [36, 35, 33, 30, 26, 21, 15, 8])
    del out
    
    
    out = bootstrapped_returns_from_episode(D, last_Vs, 0.1)
    assert out.shape == (3, D.maxT)
    assert np.allclose(out[0], [1.23, 2.3, 3, 0, 0, 0, 0, 0])
    assert np.allclose(out[1], [1.2347, 2.347, 3.47, 4.7, 7, 0, 0, 0])
    assert np.allclose(out[2], [1.2345678, 2.345678, 3.45678, 4.5678, 5.678, 6.78, 7.8, 8])
    
    with pytest.raises(AssertionError):
        bootstrapped_returns_from_segment(D, last_Vs, 0.1)

    
def test_bootstrapped_returns_from_segment():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)

    D = BatchSegment(env_spec, 5)
    D.r[0] = [1, 2, 3, 4, 5]
    D.done[0] = [False, False, False, False, False]
    D.r[1] = [1, 2, 3, 4, 5]
    D.done[1] = [False, False, True, False, False]
    D.r[2] = [1, 2, 3, 4, 5]
    D.done[2] = [True, False, False, False, True]
    
    last_Vs = torch.tensor([10, 20, 30]).unsqueeze(1)
    
    out = bootstrapped_returns_from_segment(D, last_Vs, 1.0)
    assert out.shape == (3, 5)
    assert np.allclose(out[0], [25, 24, 22, 19, 15])
    assert np.allclose(out[1], [6, 5, 3, 29, 25])
    assert np.allclose(out[2], [1, 14, 12, 9, 5])
    del out
    
    out = bootstrapped_returns_from_segment(D, last_Vs, 0.1)
    assert out.shape == (3, 5)
    assert np.allclose(out[0], [1.2346, 2.346, 3.46, 4.6, 6])
    assert np.allclose(out[1], [1.23, 2.3, 3, 4.7, 7])
    assert np.allclose(out[2], [1, 2.345, 3.45, 4.5, 5])
    
    with pytest.raises(AssertionError):
        bootstrapped_returns_from_episode(D, last_Vs, 0.1)
    
    
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
