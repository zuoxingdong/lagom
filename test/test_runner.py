from itertools import chain

import math
import numpy as np

import pytest

import gym

from lagom import RandomAgent
from lagom.envs import make_vec_env
from lagom.envs.wrappers import TimeLimit
from lagom.envs.wrappers import VecStepInfo
from lagom.runner import Trajectory
from lagom.runner import EpisodeRunner

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
        next_observation, reward, step_info = env.step(action)
        # unbatched for [reward, step_info]
        reward, step_info = map(lambda x: x[0], [reward, step_info])
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


@pytest.mark.parametrize('env_id', ['Sanity', 'CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
@pytest.mark.parametrize('num_env', [1, 3])
@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('T', [1, 5, 100])
def test_episode_runner(env_id, num_env, init_seed, T):    
    if env_id == 'Sanity':
        make_env = lambda: TimeLimit(SanityEnv())
    else:
        make_env = lambda: gym.make(env_id)
    env = make_vec_env(make_env, num_env, init_seed)
    env = VecStepInfo(env)
    agent = RandomAgent(None, env, None)
    runner = EpisodeRunner()
    
    if num_env > 1:
        with pytest.raises(AssertionError):
            D = runner(agent, env, T)
    else:
        with pytest.raises(AssertionError):
            runner(agent, env.env, T)  # must be VecStepInfo
        D = runner(agent, env, T)
        for traj in D:
            assert isinstance(traj, Trajectory)
            assert len(traj) <= env.spec.max_episode_steps
            assert traj.numpy_observations.shape == (len(traj) + 1, *env.observation_space.shape)
            if isinstance(env.action_space, gym.spaces.Discrete):
                assert traj.numpy_actions.shape == (len(traj),)
            else:
                assert traj.numpy_actions.shape == (len(traj), *env.action_space.shape)
            assert traj.numpy_rewards.shape == (len(traj),)
            assert traj.numpy_dones.shape == (len(traj), )
            assert traj.numpy_masks.shape == (len(traj), )
            assert len(traj.step_infos) == len(traj)
            if traj.completed:
                assert np.allclose(traj.observations[-1], traj.step_infos[-1]['last_observation'])
