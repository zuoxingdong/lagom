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

from lagom.envs import EnvSpec
from lagom.envs import make_gym_env
from lagom.envs import make_envs
from lagom.envs import make_vec_env
from lagom.envs.vec_env import SerialVecEnv

from lagom.agents import BaseAgent
from lagom.agents import RandomAgent

from lagom.history import Transition
from lagom.history import Trajectory
from lagom.history import Segment

from lagom.agents import StickyAgent

from lagom.runner import RollingRunner

from lagom.runner import TrajectoryRunner
from lagom.runner import SegmentRunner


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
