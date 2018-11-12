import numpy as np

import pytest

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from lagom.networks import ortho_init
from lagom.networks import BaseRNN

from .utils import make_sanity_envs

from lagom.utils import Seeder

from lagom.envs import EnvSpec
from lagom.envs import make_gym_env
from lagom.envs import make_envs
from lagom.envs import make_vec_env
from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.vec_env import ParallelVecEnv

from lagom.agents import BaseAgent
from lagom.agents import RandomAgent

from lagom.history import Transition
from lagom.history import Trajectory
from lagom.history import Segment

from lagom.history import BatchEpisode
from lagom.history import BatchSegment

from lagom.agents import StickyAgent

from lagom.runner import RollingRunner

from lagom.runner import TrajectoryRunner
from lagom.runner import SegmentRunner

from lagom.runner import EpisodeRunner
from lagom.runner import RollingSegmentRunner


@pytest.mark.parametrize('vec_env', [SerialVecEnv, ParallelVecEnv])
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_episode_runner(vec_env, env_id):
    env = make_vec_env(vec_env, make_gym_env, env_id, 3, 0)
    env_spec = EnvSpec(env)

    if env_id == 'CartPole-v1':
        sticky_action = 1
        action_shape = ()
        action_dtype = np.int32
    elif env_id == 'Pendulum-v0':
        sticky_action = [0.1]
        action_shape = env_spec.action_space.shape
        action_dtype = np.float32

    T = 30

    agent = StickyAgent(None, env_spec, sticky_action)
    runner = EpisodeRunner(None, agent, env)
    D = runner(T)

    assert D.N == 3
    assert D.maxT == max(D.Ts)

    seeder = Seeder(0)
    seed1, seed2, seed3 = seeder(3)
    env1 = make_gym_env(env_id, seed1)
    env2 = make_gym_env(env_id, seed2)
    env3 = make_gym_env(env_id, seed3)

    for n, ev in enumerate([env1, env2, env3]):
        obs = ev.reset()
        assert np.allclose(obs, D.observations[n][0])
        assert np.allclose(obs, D.numpy_observations[n, 0, ...])
        for t in range(T):
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
def test_rolling_segment_runner(vec_env, env_id):
    env = make_vec_env(vec_env, make_gym_env, env_id, 3, 0)
    env_spec = EnvSpec(env)

    if env_id == 'CartPole-v1':
        sticky_action = 1
        action_shape = ()
        action_dtype = np.int32
    elif env_id == 'Pendulum-v0':
        sticky_action = [0.1]
        action_shape = env_spec.action_space.shape
        action_dtype = np.float32

    T = 30

    agent = StickyAgent(None, env_spec, sticky_action)
    runner = RollingSegmentRunner(None, agent, env)
    D = runner(T)

    assert D.N == 3
    assert D.T == T

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

                
                

@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_rolling_runner(env_id):
    env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    env_spec = EnvSpec(env)
    
    if env_id == 'CartPole-v1':
        sticky_action = 0
    elif env_id == 'Pendulum-v0':
        sticky_action = 0.1
        
    agent = StickyAgent(None, env_spec, 0)
    runner = RollingRunner(None, agent, env)
    D = runner(20)

    sanity_env = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 3, 0)
    init_obs = sanity_env.reset()
    assert np.allclose(init_obs, D.observations[:, 0, ...])

    terminal_idx = []
    for t in range(20):
        obs, reward, done, info = sanity_env.step([0]*3)
        assert np.allclose(obs, D.observations[:, t+1, ...])
        assert np.allclose(reward, D.rewards[:, t])
        assert np.allclose(done, D.dones[:, t])

        for idx, dn in enumerate(done):
            if dn:
                terminal_idx.append([idx, t])
                assert 'terminal_observation' in D.infos[t][idx]

        assert all([len(d1) == len(d2) for d1, d2 in zip(D.infos[t], info)])


    D_idx = np.where(D.dones == True)
    D_idx = [[x, y] for x, y in zip(D_idx[0], D_idx[1])]
    assert len(D_idx) == len(terminal_idx)
    assert all([idx in D_idx for idx in terminal_idx])

    assert np.allclose(D.masks, np.logical_not(D.dones))




@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_trajectory_runner(env_id):
    env = make_vec_env(SerialVecEnv, make_gym_env, env_id, 3, 0)
    env_spec = EnvSpec(env)

    agent = RandomAgent(None, env_spec)

    runner = TrajectoryRunner(None, agent, env)
    D = runner(4)

    assert len(D) == 3
    assert all([isinstance(d, Trajectory) for d in D])
    assert all([d.T == 4 for d in D])

    # Check if s in transition is equal to s_next in previous transition
    for d in D:
        for s1, s2 in zip(d.transitions[:-1], d.transitions[1:]):
            assert np.allclose(s1.s_next, s2.s)

    # Long horizon
    D = runner(T=1000)
    for d in D:
        if d.T < 1000:
            assert d.all_done[-1] == True

        
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_segment_runner(env_id):
    env = make_vec_env(SerialVecEnv, make_gym_env, env_id, 3, 0)
    env_spec = EnvSpec(env)

    agent = RandomAgent(None, env_spec)

    runner = SegmentRunner(None, agent, env)
    D = runner(4)

    assert len(D) == 3
    assert all([isinstance(d, Segment) for d in D])
    assert all([d.T == 4 for d in D])

    # Check if s in transition is equal to s_next in previous transition
    for d in D:
        for s1, s2 in zip(d.transitions[:-1], d.transitions[1:]):
            assert np.allclose(s1.s_next, s2.s)

    # Long horizon
    D = runner(T=1000)
    for d in D:
        assert d.T == 1000
