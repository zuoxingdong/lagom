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

from lagom.runner import TrajectoryRunner
from lagom.runner import SegmentRunner


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
