import pytest
import numpy as np

import gym
from lagom.envs import RecordEpisodeStatistics
from lagom.envs import NormalizeObservation
from lagom.envs import NormalizeReward
from lagom.envs import TimeStepEnv


@pytest.mark.parametrize('env_id', ['CartPole-v0', 'Pendulum-v0'])
@pytest.mark.parametrize('deque_size', [2, 5])
def test_record_episode_statistics(env_id, deque_size):
    env = gym.make(env_id)
    env = RecordEpisodeStatistics(env, deque_size)

    for n in range(5):
        env.reset()
        assert env.episode_return == 0.0
        assert env.episode_horizon == 0
        for t in range(env.spec.max_episode_steps):
            _, _, done, info = env.step(env.action_space.sample())
            if done:
                assert 'episode' in info
                assert all([item in info['episode'] for item in ['return', 'horizon', 'time']])
                break
    assert len(env.return_queue) == deque_size
    assert len(env.horizon_queue) == deque_size
    
    
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_normalize_observation(env_id):
    env = gym.make(env_id)
    wrapped_env = NormalizeObservation(gym.make(env_id))
    unbiased = []

    env.seed(0)
    wrapped_env.seed(0)

    obs = env.reset()
    wrapped_obs = wrapped_env.reset()
    unbiased.append(obs)

    for t in range(env.spec.max_episode_steps):
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        wrapped_obs, _, wrapped_done, _ = wrapped_env.step(action)
        unbiased.append(obs)

        mean = np.mean(unbiased, 0)
        var = np.var(unbiased, 0)
        assert np.allclose(wrapped_env.obs_moments.mean, mean, atol=1e-5)
        assert np.allclose(wrapped_env.obs_moments.var, var, atol=1e-4)

        assert done == wrapped_done
        if done:
            break
            
            
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
@pytest.mark.parametrize('gamma', [0.5, 0.99])
def test_normalize_reward(env_id, gamma):
    env = gym.make(env_id)
    wrapped_env = NormalizeReward(gym.make(env_id), gamma=gamma)
    unbiased = []

    env.seed(0)
    wrapped_env.seed(0)

    for n in range(10):
        env.reset()
        wrapped_env.reset()
        G = 0.0
        for t in range(env.spec.max_episode_steps):
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            _, wrapped_reward, wrapped_done, _ = wrapped_env.step(action)
            assert done == wrapped_done

            G = reward + gamma*G
            unbiased.append(G)

            if done:
                break

            mean = np.mean(unbiased, 0)
            var = np.var(unbiased, 0)
            assert wrapped_env.all_returns == G

            assert np.allclose(wrapped_env.reward_moments.mean, mean, atol=1e-4)
            assert np.allclose(wrapped_env.reward_moments.var, var, atol=1e-3)
            
            
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_timestep_env(env_id):
    env = gym.make(env_id)
    wrapped_env = TimeStepEnv(gym.make(env_id))

    env.seed(0)
    wrapped_env.seed(0)

    obs = env.reset()
    timestep = wrapped_env.reset()
    assert timestep.first()
    assert np.allclose(timestep.observation, obs)

    for t in range(env.spec.max_episode_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        timestep = wrapped_env.step(action)
        assert np.allclose(timestep.observation, obs)
        assert timestep.reward == reward
        assert timestep.done == done
        assert timestep.info == info
        if done:
            assert timestep.last()
            if 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
                assert timestep.time_limit()
            else:
                assert timestep.terminal()
            break
        else:
            assert timestep.mid()
