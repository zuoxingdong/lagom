from itertools import chain

import math
import numpy as np

import pytest

import gym

from lagom.agents import RandomAgent
from lagom.runner import BatchHistory
from lagom.runner import EpisodeRunner
from lagom.runner import RollingSegmentRunner
from lagom.envs import make_vec_env

from .sanity_env import SanityEnv


@pytest.mark.parametrize('num_env', [1, 3])
@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('mode', ['serial', 'parallel'])
@pytest.mark.parametrize('T', [1, 5, 100])
def test_batch_history(num_env, init_seed, mode, T):
    make_env = lambda: SanityEnv()
    env = make_vec_env(make_env, num_env, init_seed, mode)
    D = BatchHistory(env)
    assert D.env == env
    assert D.N == num_env
    assert isinstance(D.s, list) and len(D.s) == num_env
    assert isinstance(D.a, list) and len(D.a) == num_env
    assert isinstance(D.r, list) and len(D.r) == num_env
    assert isinstance(D.done, list) and len(D.done) == num_env
    assert isinstance(D.info, list) and len(D.info) == num_env
    assert isinstance(D.batch_info, list) and len(D.batch_info) == 0
    assert D.T == 0
    assert not any(D.stops)

    observations = env.reset()
    for t in range(T):
        actions = [env.action_space.sample() for _ in range(env.num_env)]
        next_observations, rewards, dones, infos = env.step(actions)
        D.add(observations, actions, rewards, dones, infos, {})
        observations = next_observations
    for n, done in enumerate(dones):
        if not done:
            D.info[n][-1][-1]['last_observation'] = observations[n]

    assert D.N == num_env and D.T == T
    assert all([sum(t) == D.T for t in D.Ts])
    
    def _flat(items):
        return list(chain.from_iterable(items))
    def _check(items, raw_items, pad):
        for item, raw_item in zip(items, _flat(raw_items)):
            if len(raw_item) < item.shape[0]:
                assert np.allclose(item,  raw_item + [pad]*(item.shape[0] - len(raw_item)))
            else:
                assert np.allclose(item, raw_item)
            #print(item, raw_item)
    def _check_batch(items, raw_items):
        for item, raw_item in zip(items, raw_items):
            #print(item, raw_item)
            assert np.allclose(item, _flat(raw_item))
    _check(D.observations, D.s, 0.0)
    _check_batch(D.batch_observations, D.s)
    for last_observation, done, info in zip(D.last_observations, _flat(D.done), _flat(D.info)):
        if done[-1]:
            assert np.allclose(last_observation, info[-1]['terminal_observation'])
        else:
            assert np.allclose(last_observation, info[-1]['last_observation'])
    if D.terminal_observations is not None:
        assert D.terminal_observations.shape[0] == sum([done[-1] for done in list(chain.from_iterable(D.done))])
    _check(D.actions, D.a, 0.0)
    _check_batch(D.batch_actions, D.a)
    _check(D.rewards, D.r, 0.0)
    _check_batch(D.batch_rewards, D.r)
    _check(D.dones, D.done, True)
    _check_batch(D.batch_dones, D.done)
    assert np.allclose(D.masks, np.logical_not(D.dones))
    assert np.allclose(D.batch_masks, np.logical_not(D.batch_dones))
    for validity_mask, T in zip(D.validity_masks, D.Ts_flat):
        assert np.allclose(validity_mask[:T], 1.0)
        assert np.allclose(validity_mask[T:], 0.0)
    for batch_validity_mask, T in zip(D.batch_validity_masks, D.Ts):
        assert np.allclose(batch_validity_mask[:sum(T)], 1.0)
        assert np.allclose(batch_validity_mask[sum(T):], 0.0)


@pytest.mark.parametrize('env_id', ['Sanity', 'CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
@pytest.mark.parametrize('num_env', [1, 3])
@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('mode', ['serial', 'parallel'])
@pytest.mark.parametrize('T', [1, 5, 100])
def test_episode_runner(env_id, num_env, init_seed, mode, T):
    if env_id == 'Sanity':
        make_env = lambda: SanityEnv()
    else:
        make_env = lambda: gym.make(env_id)
    env = make_vec_env(make_env, num_env, init_seed, mode)
    agent = RandomAgent(None, env, None)
    runner = EpisodeRunner()
    D = runner(agent, env, T)
    assert sum(D.num_traj) == env.num_env
    assert D.observations.shape[0] == env.num_env
    assert np.allclose(D.observations, D.batch_observations)
    assert D.last_observations.shape[0] == num_env
    assert D.actions.shape[0] == env.num_env
    assert np.allclose(D.actions, D.batch_actions)
    assert D.rewards.shape[0] == env.num_env
    assert np.allclose(D.rewards, D.batch_rewards)
    assert D.dones.shape[0] == env.num_env
    assert np.allclose(D.dones, D.batch_dones)
    assert D.masks.shape[0] == env.num_env
    assert np.allclose(D.masks, D.batch_masks)
    assert D.validity_masks.shape[0] == env.num_env
    assert np.allclose(D.validity_masks, D.batch_validity_masks)

    
@pytest.mark.parametrize('env_id', ['Sanity', 'CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
@pytest.mark.parametrize('num_env', [1, 3])
@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('mode', ['serial', 'parallel'])
@pytest.mark.parametrize('T', [1, 5, 100])
def test_rolling_segment_runner(env_id, num_env, init_seed, mode, T):
    if env_id == 'Sanity':
        make_env = lambda: SanityEnv()
    else:
        make_env = lambda: gym.make(env_id)
    env = make_vec_env(make_env, num_env, init_seed, mode)
    agent = RandomAgent(None, env, None)
    runner = RollingSegmentRunner()
    D = runner(agent, env, T)
    assert sum(D.num_traj) >= env.num_env
    assert D.observations.shape[0] >= env.num_env
    assert D.batch_observations.shape[:2] == (num_env, T)
    assert D.last_observations.shape[0] == sum(D.num_traj)
    assert D.actions.shape[0] >= env.num_env
    assert D.batch_actions.shape[:2] == (num_env, T)
    assert D.rewards.shape[0] >= env.num_env
    assert D.batch_rewards.shape[:2] == (num_env, T)
    assert D.dones.shape[0] >= env.num_env
    assert D.batch_dones.shape[:2] == (num_env, T)
    assert D.masks.shape[0] >= env.num_env
    assert D.batch_masks.shape[:2] == (num_env, T)
    assert D.validity_masks.shape[0] >= env.num_env
    assert D.batch_validity_masks.shape[:2] == (num_env, T)
