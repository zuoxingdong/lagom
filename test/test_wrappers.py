import numpy as np

import gym

from lagom.envs.spaces import Discrete
from lagom.envs.spaces import Box

from lagom.envs import Env
from lagom.envs import make_gym_env

from lagom.envs.wrappers import Wrapper
from lagom.envs.wrappers import GymWrapper
from lagom.envs.wrappers import FrameStack
from lagom.envs.wrappers import RewardScale

# FlattenDictWrapper requires Mujoco, so omitted from test


def test_gym_wrapper():
    gym_env = gym.make('CartPole-v1')
    env = GymWrapper(gym_env)
    assert isinstance(env, GymWrapper)
    assert isinstance(env, Wrapper)
    assert isinstance(env.env, gym.Wrapper)
    assert env.reset().shape == (4,)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)
    assert isinstance(env.unwrapped, gym.Env)
    assert len(env.step(env.action_space.sample())) == 4
    assert env.seed(3) == [3]
    assert env.T == 500
    assert env.max_episode_reward == 475.0
    assert env.reward_range == (-float('inf'), float('inf'))

    del gym_env
    del env

    gym_env = gym.make('Pendulum-v0')
    env = GymWrapper(gym_env)
    assert isinstance(env, GymWrapper)
    assert isinstance(env, Wrapper)
    assert isinstance(env.env, gym.Wrapper)
    assert env.reset().shape == (3,)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Box)
    assert isinstance(env.unwrapped, gym.Env)
    assert len(env.step(env.action_space.sample())) == 4
    assert env.seed(3) == [3]
    assert env.T == 200
    assert env.max_episode_reward is None
    assert env.reward_range == (-float('inf'), float('inf'))

    del gym_env
    del env
    
    
def test_frame_stack():
    env = make_gym_env(env_id='CartPole-v1', seed=1)
    env = FrameStack(env, num_stack=4)
    assert isinstance(env, FrameStack)
    assert isinstance(env, Env)
    assert env.num_stack == 4
    assert env.observation_space.shape == (4, 4)
    assert isinstance(env.stack_buffer, np.ndarray)
    assert env.stack_buffer.shape == (4, 4)
    assert np.all(env.stack_buffer == 0.0)
    assert env.stack_buffer.dtype == np.float32
    assert env.reset().shape == (4, 4)
    obs = env.step(0)[0]
    assert obs[:, 0].sum() != 0.0
    assert obs[:, 1].sum() != 0.0
    assert np.all(obs[:, 2:] == 0.0)
    assert np.any(obs[:, 0] != obs[:, 1])
    obs = env.step(1)[0]
    obs = env.step(1)[0]
    assert np.allclose(obs[:, -1], [0.03073904, 0.00145001, -0.03088818, -0.03131252])
    assert np.allclose(obs[:, 2], [0.03076804, -0.19321568, -0.03151444, 0.25146705])
    obs = env.step(1)[0]
    assert np.allclose(obs[:, -1], [0.03076804, -0.19321568, -0.03151444, 0.25146705])

    
def test_reward_scale():
    env = make_gym_env(env_id='CartPole-v1', seed=0)
    env = RewardScale(env, scale=0.02)
    env.reset()
    observation, reward, done, info = env.step(env.action_space.sample())
    assert reward == 0.02
