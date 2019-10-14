import numpy as np

import pytest

from lagom.utils import Seeder

import gym
from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import Tuple
from gym.spaces import Dict

from lagom.envs import RecordEpisodeStatistics
from lagom.envs import NormalizeObservation
from lagom.envs import NormalizeReward
from lagom.envs import TimeStepEnv
from lagom.envs import make_vec_env
from lagom.envs import VecEnv


@pytest.mark.parametrize('env_id', ['CartPole-v0', 'Pendulum-v0'])
@pytest.mark.parametrize('num_env', [1, 3, 5])
def test_vec_env(env_id, num_env):
    def make_env():
        return gym.make(env_id)
    base_env = make_env()
    list_make_env = [make_env for _ in range(num_env)]
    env = VecEnv(list_make_env)
    assert isinstance(env, VecEnv)
    assert len(env) == num_env
    assert len(list(env)) == num_env
    assert env.observation_space == base_env.observation_space
    assert env.action_space == base_env.action_space
    assert env.reward_range == base_env.reward_range
    assert env.spec.id == base_env.spec.id
    obs = env.reset()
    assert isinstance(obs, list) and len(obs) == num_env
    assert all([x in env.observation_space for x in obs])
    actions = [env.action_space.sample() for _ in range(num_env)]
    observations, rewards, dones, infos = env.step(actions)
    assert isinstance(observations, list) and len(observations) == num_env
    assert isinstance(rewards, list) and len(rewards) == num_env
    assert isinstance(dones, list) and len(dones) == num_env
    assert isinstance(infos, list) and len(infos) == num_env
    env.close()
    assert env.closed


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
    

@pytest.mark.parametrize('env_id', ['CartPole-v0', 'Pendulum-v0'])
@pytest.mark.parametrize('num_env', [1, 3, 5])
@pytest.mark.parametrize('init_seed', [0, 10])
def test_make_vec_env(env_id, num_env, init_seed):
    def make_env():
        return gym.make(env_id)
    env = make_vec_env(make_env, num_env, init_seed)
    assert isinstance(env, VecEnv)
    seeds = [x.keywords['seed'] for x in env.list_make_env]
    seeder = Seeder(init_seed)
    assert seeds == seeder(num_env)
