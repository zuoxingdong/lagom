import numpy as np

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.distributions import Normal
from torch.distributions import Independent

from lagom.envs import make_gym_env
from lagom.envs import EnvSpec
from lagom.envs import make_vec_env
from lagom.envs.vec_env import SerialVecEnv

from lagom.policies import RandomPolicy
from lagom.policies import CategoricalHead
from lagom.policies import DiagGaussianHead


def test_random_policy():
    env = make_gym_env('Pendulum-v0', 0)
    env_spec = EnvSpec(env)
    policy = RandomPolicy(None, env_spec)
    out = policy(env.reset())
    assert isinstance(out, dict)
    assert 'action' in out and out['action'].shape == (1,)

    venv = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v0', 3, 0)
    env_spec = EnvSpec(venv)
    policy = RandomPolicy(None, env_spec)
    out = policy(env.reset())
    assert isinstance(out, dict)
    assert 'action' in out and len(out['action']) == 3 and isinstance(out['action'][0], int)


def test_diag_gaussian_head():
    with pytest.raises(AssertionError):
        env = make_gym_env('CartPole-v1', 0)
        env_spec = EnvSpec(env)
        DiagGaussianHead(None, None, 30, env_spec)

    env = make_gym_env('Pendulum-v0', 0)
    env_spec = EnvSpec(env)
    head = DiagGaussianHead(None, None, 30, env_spec)
    assert head.feature_dim == 30
    assert isinstance(head.mean_head, nn.Linear)
    assert isinstance(head.logstd_head, nn.Parameter)
    assert head.mean_head.in_features == 30 and head.mean_head.out_features == 1
    assert list(head.logstd_head.shape) == [1]
    assert torch.eq(head.logstd_head, torch.tensor(-0.510825624))
    dist = head(torch.randn(3, 30))
    assert isinstance(dist, Independent) and isinstance(dist.base_dist, Normal)
    assert list(dist.batch_shape) == [3]
    action = dist.sample()
    assert list(action.shape) == [3, 1]

    head = DiagGaussianHead(None, None, 30 , env_spec, std_style='softplus')
    dist = head(torch.randn(3, 30))
    action = dist.sample()
    assert list(action.shape) == [3, 1]
    assert torch.eq(head.logstd_head, torch.tensor(-0.19587036834631966))
    
    head = DiagGaussianHead(None, None, 30 , env_spec, std_style='sigmoidal')
    assert torch.eq(head.logstd_head, torch.tensor(-0.871222446472449))

    head = DiagGaussianHead(None, None, 30, env_spec, std_state_dependent=True)
    dist = head(torch.randn(3, 30))
    action = dist.sample()
    assert list(action.shape) == [3, 1]

    head = DiagGaussianHead(None, None, 30, env_spec, constant_std=0.3)
    dist = head(torch.randn(3, 30))
    action = dist.sample()
    assert list(action.shape) == [3, 1]
    assert not head.logstd_head.requires_grad
    assert torch.eq(head.logstd_head, torch.tensor([-1.2039728]))


def test_categorical_head():
    with pytest.raises(AssertionError):
        env = make_gym_env('Pendulum-v0', 0)
        env_spec = EnvSpec(env)
        CategoricalHead(None, None, 30, env_spec)

    env = make_gym_env('CartPole-v1', 0)
    env_spec = EnvSpec(env)
    head = CategoricalHead(None, None, 30, env_spec)
    assert head.feature_dim == 30
    assert isinstance(head.action_head, nn.Linear)
    assert head.action_head.in_features == 30 and head.action_head.out_features == 2
    dist = head(torch.randn(3, 30))
    assert isinstance(dist, Categorical)
    assert list(dist.batch_shape) == [3]
    assert list(dist.probs.shape) == [3, 2]
    action = dist.sample()
    assert action.shape == (3,)
