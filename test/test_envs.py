import numpy as np

import gym

from lagom import Seeder

from lagom.envs.spaces import Discrete
from lagom.envs.spaces import Box

from lagom.envs import Env
from lagom.envs import EnvSpec

from lagom.envs.wrappers import Wrapper
from lagom.envs.wrappers import GymWrapper

from lagom.envs import make_gym_env
from lagom.envs import make_envs
from lagom.envs import make_vec_env

from lagom.envs.vec_env import VecEnv
from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.vec_env import ParallelVecEnv


def test_make_gym_env():
    env = make_gym_env(env_id='CartPole-v1', seed=0, monitor=False)
    assert isinstance(env, Env)
    assert not isinstance(env, gym.Env)
    assert isinstance(env, Wrapper)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)
    env_spec = EnvSpec(env)
    assert env_spec.control_type == 'Discrete'
    assert env_spec.T == 500
    assert env_spec.max_episode_reward == 475.0
    assert env_spec.reward_range == (-float('inf'), float('inf'))

    assert env.reset().shape == (4,)
    assert len(env.step(env.action_space.sample())) == 4

    del env
    del env_spec

    # Pendulum, continuous
    # do not test redundant part
    env = make_gym_env('Pendulum-v0', seed=0)
    assert isinstance(env, Env)
    env_spec = EnvSpec(env)
    assert isinstance(env_spec.action_space, Box)
    assert env_spec.T == 200
    assert env_spec.control_type == 'Continuous'

    assert env.reset().shape == (3,)
    assert len(env.step(env.action_space.sample())) == 4

def test_make_envs():
    list_make_env = make_envs(make_env=make_gym_env, env_id='Pendulum-v0', num_env=3, init_seed=1)
    assert len(list_make_env) == 3
    assert list_make_env[0] != list_make_env[1] and list_make_env[0] != list_make_env[2]

    # Test if the seedings are correct
    seeder = Seeder(init_seed=1)
    seeds = seeder(3)
    for make_env, seed in zip(list_make_env, seeds):
        assert make_env.keywords['seed'] == seed
    env = list_make_env[0]()
    raw_env = gym.make('Pendulum-v0')
    raw_env.seed(seeds[0])
    assert np.allclose(env.reset(), raw_env.reset())
    
def test_make_vec_env():
    venv1 = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 5, 1)
    venv2 = make_vec_env(ParallelVecEnv, make_gym_env, 'CartPole-v1', 5, 1)
    assert isinstance(venv1, VecEnv) and isinstance(venv1, SerialVecEnv)
    assert isinstance(venv2, VecEnv) and isinstance(venv2, ParallelVecEnv)
    assert venv1.num_env == venv2.num_env
    assert venv1.observation_space == venv2.observation_space
    assert venv1.action_space == venv2.action_space
    assert venv1.reward_range == venv2.reward_range
    assert venv1.T == venv2.T
    o1 = venv1.reset()
    o2 = venv2.reset()
    # Two environments should have same random seeds, then same results under same actions
    assert np.allclose(o1, o2)
    a = [1]*5
    o1, r1, d1, _ = venv1.step(a)
    o2, r2, d2, _ = venv2.step(a)
    assert np.allclose(o1, o2)
    assert np.allclose(r1, r2)
    assert np.allclose(d1, d2)
    assert not venv1.closed
    venv1.close()
    assert venv1.closed
    assert not venv2.closed
    venv2.close()
    assert venv2.closed
