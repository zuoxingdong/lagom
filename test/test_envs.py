import numpy as np

import gym

from lagom.envs import Env
from lagom.envs import make_gym_env
from lagom.envs import make_envs

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
