import numpy as np

from .utils import make_sanity_envs

from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.spaces import Box, Discrete


def test_sanity_check_env():
    env = SerialVecEnv(make_sanity_envs([2, 3]))
    assert isinstance(env.observation_space, Box) and isinstance(env.action_space, Discrete)
    obs = env.reset()
    assert np.allclose(obs, [[0.01], [0.01]])
    obs, reward, done, info = env.step([0, 1])
    assert np.allclose(obs, [[1.01], [1.01]])
    assert np.allclose(reward, [0.1, 0.1])
    assert np.allclose(done, [False, False])
    assert all(len(i) == 0 for i in info)

    obs, reward, done, info = env.step([1, 0])
    assert np.allclose(obs, [[2.01], [2.01]])
    assert np.allclose(reward, [0.2, 0.2])
    assert np.allclose(done, [True, False])
    assert info[0]['init_observation'] == [0.01] and len(info[1]) == 0

    obs, reward, done, info = env.step([1, 1])
    assert np.allclose(obs, [[1.01], [3.01]])
    assert np.allclose(reward, [0.1, 0.3])
    assert np.allclose(done, [False, True])
    assert len(info[0]) == 0 and info[1]['init_observation'] == [0.01]

    obs, reward, done, info = env.step([0, 0])
    assert np.allclose(obs, [[2.01], [1.01]])
    assert np.allclose(reward, [0.2, 0.1])
    assert np.allclose(done, [True, False])
    assert info[0]['init_observation'] == [0.01] and len(info[1]) == 0
