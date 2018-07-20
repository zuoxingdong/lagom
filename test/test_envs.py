import numpy as np

import gym

from lagom.envs import Env
from lagom.envs import GymEnv
from lagom.envs import make_gym_env
from lagom.envs import make_envs

from lagom.envs.wrappers import StackObservation

from lagom import Seeder


class TestEnvs(object):
    def test_make_gym_env(self):
        env = make_gym_env(env_id='Pendulum-v0', seed=1)
        assert isinstance(env, Env)

    def test_make_envs(self):
        list_make_env = make_envs(make_env=make_gym_env, env_id='Pendulum-v0', num_env=3, init_seed=1)
        assert len(list_make_env) == 3
        assert list_make_env[0] != list_make_env[1] and list_make_env[0] != list_make_env[2]

        # Test if the seedings are correct
        seeder = Seeder(init_seed=1)
        seeds=  seeder(3)
        for make_env, seed in zip(list_make_env, seeds):
            assert make_env.keywords['seed'] == seed
        env = list_make_env[0]()
        raw_env = gym.make('Pendulum-v0')
        raw_env.seed(seeds[0])
        assert np.allclose(env.reset(), raw_env.reset())


class TestWrapper(object):
    def test_stackobservation(self):
        env = gym.make('Pendulum-v0')
        env = GymEnv(env)
        env.seed(0)
        env = StackObservation(env, 3)

        raw_env = gym.make('Pendulum-v0')
        raw_env.seed(0)

        init_env = env.reset()
        init_raw_env = raw_env.reset()

        # Check initial obsevations
        assert np.allclose(init_env[..., 0], init_raw_env)
        assert np.allclose(init_env[..., 1:], 0.0)

        # Check one step
        action = raw_env.action_space.sample()
        env_obs, _, _, _ = env.step(action)
        raw_env_obs, _, _, _ = raw_env.step(action)

        assert np.allclose(env_obs[..., 0], raw_env_obs)
        assert np.allclose(env_obs[..., 1], init_raw_env)
        assert np.allclose(env_obs[..., 2], 0.0)

        # Check rolling effect of stack observation
        for _ in range(2):
            action = raw_env.action_space.sample()
            env_obs, _, _, _ = env.step(action)
            raw_env_obs, _, _, _ = raw_env.step(action)

        assert np.allclose(env_obs[..., 0], raw_env_obs)
        assert not np.allclose(env_obs[..., 2], init_raw_env)
