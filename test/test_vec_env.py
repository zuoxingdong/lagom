import pytest

import numpy as np

from lagom.envs import make_gym_env
from lagom.envs import make_vec_env

from lagom.envs import EnvSpec

from lagom.envs.spaces import Space
from lagom.envs.spaces import Discrete
from lagom.envs.spaces import Box

from lagom.envs.vec_env import VecEnv
from lagom.envs.vec_env import VecEnvWrapper

from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.vec_env import ParallelVecEnv
from lagom.envs.vec_env import VecStandardize


@pytest.mark.parametrize('vec_env_class', [(0, SerialVecEnv), (1, ParallelVecEnv)])
def test_vec_env(vec_env_class):
    # unpack class
    v_id, vec_env_class = vec_env_class
    
    venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1, True)
    assert isinstance(venv, VecEnv)
    assert v_id in [0, 1]
    if v_id == 0:
        isinstance(venv, SerialVecEnv)
    elif v_id == 1:
        assert isinstance(venv, ParallelVecEnv)
    
    assert venv.num_env == 5
    assert not venv.closed and venv.viewer is None
    assert venv.unwrapped is venv
    assert isinstance(venv.observation_space, Box)
    assert isinstance(venv.action_space, Discrete)
    assert venv.T == 500
    assert venv.max_episode_reward == 475.0
    assert venv.reward_range == (-float('inf'), float('inf'))
    obs = venv.reset()
    assert len(obs) == 5
    assert np.asarray(obs).shape == (5, 4)
    assert all([not np.allclose(obs[0], obs[i]) for i in [1, 2, 3, 4]])
    a = [1]*5
    obs, rewards, dones, infos = venv.step(a)
    assert all([len(item) == 5 for item in [obs, rewards, dones, infos]])
    assert all([not np.allclose(obs[0], obs[i]) for i in [1, 2, 3, 4]])

    # EnvSpec
    env_spec = EnvSpec(venv)
    assert isinstance(env_spec.action_space, Discrete)
    assert isinstance(env_spec.observation_space, Box)
    assert env_spec.control_type == 'Discrete'
    assert env_spec.T == 500
    assert env_spec.max_episode_reward == 475.0
    assert env_spec.reward_range == (-float('inf'), float('inf'))

    venv.close()
    assert venv.closed

@pytest.mark.parametrize('vec_env_class', [SerialVecEnv, ParallelVecEnv])
def test_vec_standardize(vec_env_class):
    venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1, True)
    venv = VecStandardize(venv, 
                          use_obs=True, 
                          use_reward=True, 
                          clip_obs=10., 
                          clip_reward=10., 
                          gamma=0.99, 
                          eps=1e-8)
    assert isinstance(venv, VecEnvWrapper) and isinstance(venv, VecStandardize)
    obs = venv.reset()
    assert not np.allclose(venv.obs_runningavg.mu, 0.0)
    assert not np.allclose(venv.obs_runningavg.sigma, 0.0)
    a = [1]*5
    [venv.step(a) for _ in range(20)]
    assert venv.obs_runningavg.N == 5 + 5*20
    assert venv.reward_runningavg.N == 5*20
    assert not np.allclose(venv.obs_runningavg.mu, 0.0)
    assert not np.allclose(venv.obs_runningavg.sigma, 0.0)
    running_avg = venv.running_averages
    assert isinstance(running_avg, dict)
    assert len(running_avg) == 2 and 'obs_avg' in running_avg and 'r_avg' in running_avg
    assert 'mu' in running_avg['obs_avg'] and 'sigma' in running_avg['obs_avg']
    assert not np.allclose(running_avg['obs_avg']['mu'], 0.0)
    assert not np.allclose(running_avg['obs_avg']['sigma'], 0.0)
    assert 'mu' not in running_avg['r_avg']
    assert 'sigma' in running_avg['r_avg']
    assert not np.allclose(running_avg['r_avg']['sigma'], 0.0)

    del venv, obs, a

    # other settings: clipping
    venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1, True)
    venv = VecStandardize(venv, 
                          use_obs=True, 
                          use_reward=True, 
                          clip_obs=0.01, 
                          clip_reward=0.0001, 
                          gamma=0.99, 
                          eps=1e-8)
    obs = venv.reset()
    assert np.allclose(np.abs(np.asarray(obs)), 0.01)

    running_avg = venv.running_averages
    assert isinstance(running_avg, dict)
    assert len(running_avg) == 2 and 'obs_avg' in running_avg and 'r_avg' in running_avg
    assert 'mu' in running_avg['obs_avg'] and 'sigma' in running_avg['obs_avg']
    assert not np.allclose(running_avg['obs_avg']['mu'], 0.0)
    assert not np.allclose(running_avg['obs_avg']['sigma'], 0.0)
    assert 'mu' not in running_avg['r_avg']
    assert 'sigma' in running_avg['r_avg']
    assert running_avg['r_avg']['sigma'] is None

    a = [1]*5
    obs, rewards, _, _ = venv.step(a)
    assert rewards.max() == 0.0001

    del venv, obs, a

    # other settings: turn off use_obs
    venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1, True)
    venv = VecStandardize(venv, 
                          use_obs=False, 
                          use_reward=False, 
                          clip_obs=0.001, 
                          clip_reward=0.0001, 
                          gamma=0.99, 
                          eps=1e-8)
    obs = venv.reset()
    assert np.asarray(obs).max() > 0.001
    a = [1]*5
    obs, rewards, _, _ = venv.step(a)
    assert np.asarray(rewards).max() >= 0.0001

    del venv, obs, a

    # other settings: gamma
    venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1, True)
    with pytest.raises(AssertionError):
        venv = VecStandardize(venv, 
                              use_obs=False, 
                              use_reward=False, 
                              clip_obs=0.001, 
                              clip_reward=0.0001, 
                              gamma=1.0,  # not allowed
                              eps=1e-8)

    del venv

    # other settings: constant value 

    venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1, True)
    venv = VecStandardize(venv, 
                          use_obs=True, 
                          use_reward=True, 
                          clip_obs=10., 
                          clip_reward=10., 
                          gamma=0.99, 
                          eps=1e-8, 
                          constant_obs_mean=np.array([5.]*4),
                          constant_obs_std=np.array([1.]*4), 
                          constant_reward_std=np.array(1000))

    obs = venv.reset()
    assert obs.min() < -4.0
    a = [1]*5
    obs, rewards, _, _ = venv.step(a)
    assert rewards.min() <= 0.01
    
@pytest.mark.parametrize('rolling', [True, False])
def test_equivalence_vec_env(rolling):
    venv1 = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 5, 1, rolling)
    venv2 = make_vec_env(ParallelVecEnv, make_gym_env, 'CartPole-v1', 5, 1, rolling)
    assert venv1.observation_space == venv2.observation_space
    assert venv1.action_space == venv2.action_space
    assert venv1.num_env == venv2.num_env
    obs1 = venv1.reset()
    obs2 = venv2.reset()
    assert np.allclose(obs1, obs2)
    a = [1]*5
    obs1, rewards1, dones1, _ = venv1.step(a)
    obs2, rewards2, dones2, _ = venv2.step(a)
    assert np.allclose(obs1, obs2)
    assert np.allclose(rewards1, rewards2)
    assert np.allclose(dones1, dones2)

@pytest.mark.parametrize('vec_env_class', [SerialVecEnv, ParallelVecEnv])
def test_rolling(vec_env_class):
    venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1, rolling=False)
    venv.reset()
    for _ in range(100):
        observations, rewards, dones, infos = venv.step([venv.action_space.sample()]*5)
    assert all([len(x) == 5 for x in [observations, rewards, dones, infos]])
    assert all([x == [None]*5 for x in [observations, rewards, dones, infos]])
    venv.reset()
    result = venv.step([venv.action_space.sample()]*5)
    assert all([None not in result[i] for i in [1, 2, 3]])
