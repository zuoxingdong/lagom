import numpy as np

import pytest

from lagom.utils import Seeder

import gym
from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import Tuple
from gym.spaces import Dict

from lagom.envs import flatdim
from lagom.envs import flatten
from lagom.envs import unflatten
from lagom.envs import SerialVecEnv
from lagom.envs import ParallelVecEnv
from lagom.envs import make_vec_env
from lagom.envs import make_atari
from lagom.envs.wrappers import get_wrapper
from lagom.envs.wrappers import get_all_wrappers
from lagom.envs.wrappers import ClipAction
from lagom.envs.wrappers import ClipReward
from lagom.envs.wrappers import SignClipReward
from lagom.envs.wrappers import FlattenObservation
from lagom.envs.wrappers import FrameStack
from lagom.envs.wrappers import GrayScaleObservation
from lagom.envs.wrappers import ResizeObservation
from lagom.envs.wrappers import ScaleReward
from lagom.envs.wrappers import ScaledFloatFrame
from lagom.envs.wrappers import TimeAwareObservation
from lagom.envs.wrappers import VecMonitor
from lagom.envs.wrappers import VecStandardizeObservation
from lagom.envs.wrappers import VecStandardizeReward


def test_space_utils():
    # Box
    box = Box(-1.0, 1.0, shape=[2, 3], dtype=np.float32)
    sample = box.sample()
    assert flatdim(box) == 2*3
    assert flatten(box, sample).shape == (2*3,)
    assert np.allclose(sample, unflatten(box, flatten(box, sample)))

    x = np.array([[1.0, 1.0], [1.0, 1.0]])
    box = Box(low=-x, high=x, dtype=np.float32)
    sample = box.sample()
    assert flatdim(box) == 2*2
    assert flatten(box, sample).shape == (2*2,)
    assert np.allclose(sample, unflatten(box, flatten(box, sample)))

    # Discrete
    discrete = Discrete(5)
    sample = discrete.sample()
    assert flatdim(discrete) == 5
    assert flatten(discrete, sample).shape == (5,)
    assert sample == unflatten(discrete, flatten(discrete, sample))

    # Tuple
    S = Tuple([Discrete(5), 
               Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32), 
               Dict({'success': Discrete(2), 'velocity': Box(-1, 1, shape=(1, 3), dtype=np.float32)})])
    sample = S.sample()
    assert flatdim(S) == 5+2*3+2+3
    assert flatten(S, sample).shape == (16,)
    _sample = unflatten(S, flatten(S, sample))
    assert sample[0] == _sample[0]
    assert np.allclose(sample[1], _sample[1])
    assert sample[2]['success'] == _sample[2]['success']
    assert np.allclose(sample[2]['velocity'], _sample[2]['velocity'])

    # Dict
    D0 = Dict({'position': Box(-100, 100, shape=(3,), dtype=np.float32), 
               'velocity': Box(-1, 1, shape=(4,), dtype=np.float32)})
    D = Dict({'sensors': D0, 'score': Discrete(100)})
    sample = D.sample()
    assert flatdim(D) == 3+4+100
    assert flatten(D, sample).shape == (107,)
    _sample = unflatten(D, flatten(D, sample))
    assert sample['score'] == _sample['score']
    assert np.allclose(sample['sensors']['position'], _sample['sensors']['position'])
    assert np.allclose(sample['sensors']['velocity'], _sample['sensors']['velocity'])
    
    
@pytest.mark.parametrize('env_id', ['Pong', 'Breakout', 'SpaceInvaders'])
def test_make_atari(env_id):
    env = make_atari(env_id)
    assert env.observation_space.shape == (4, 84, 84)
    assert np.allclose(env.observation_space.low, 0)
    assert np.allclose(env.observation_space.high, 255)
    obs = env.reset()
    for _ in range(200):
        obs, reward, done, info = env.step(env.action_space.sample())
        obs = np.asarray(obs)
        assert obs.shape == (4, 84, 84)
        assert obs.max() <= 255
        assert obs.min() >= 0
        if done:
            break


@pytest.mark.parametrize('vec_env_class', [SerialVecEnv, ParallelVecEnv])
@pytest.mark.parametrize('env_id', ['CartPole-v0', 'Pendulum-v0'])
@pytest.mark.parametrize('num_env', [1, 3, 5])
def test_vec_env(vec_env_class, env_id, num_env):
    def make_env():
        return gym.make(env_id)
    base_env = make_env()
    list_make_env = [make_env for _ in range(num_env)]
    env = vec_env_class(list_make_env)
    assert isinstance(env, (SerialVecEnv, ParallelVecEnv))
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
@pytest.mark.parametrize('num_env', [1, 3, 5])
@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('mode', ['serial', 'parallel'])
def test_make_vec_env(env_id, num_env, init_seed, mode):
    def make_env():
        return gym.make(env_id)
    env = make_vec_env(make_env, num_env, init_seed, mode)
    if mode == 'serial':
        assert isinstance(env, SerialVecEnv)
    else:
        assert isinstance(env, ParallelVecEnv)
    seeds = [x.keywords['seed'] for x in env.list_make_env]
    seeder = Seeder(init_seed)
    assert seeds == seeder(num_env)
    
    
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
@pytest.mark.parametrize('num_env', [1, 3, 5])
@pytest.mark.parametrize('init_seed', [0, 10])
def test_equivalence_vec_env(env_id, num_env, init_seed):
    make_env = lambda: gym.make(env_id)

    env1 = make_vec_env(make_env, num_env, init_seed, mode='serial')
    env2 = make_vec_env(make_env, num_env, init_seed, mode='parallel')

    assert env1.observation_space == env2.observation_space
    assert env1.action_space == env2.action_space
    assert len(env1) == len(env2)
    obs1 = env1.reset()
    obs2 = env2.reset()
    assert np.allclose(obs1, obs2)

    for _ in range(20):
        actions = [env1.action_space.sample() for _ in range(len(env1))]
        obs1, rewards1, dones1, _ = env1.step(actions)
        obs2, rewards2, dones2, _ = env2.step(actions)
        assert np.allclose(obs1, obs2)
        assert np.allclose(rewards1, rewards2)
        assert np.allclose(dones1, dones2)

    
def test_clip_action():
    # mountaincar: action-based rewards
    env = gym.make('MountainCarContinuous-v0')
    clipped_env = ClipAction(env)

    env.reset()
    clipped_env.reset()

    action = [10000.]

    _, reward, _, _ = env.step(action)
    _, clipped_reward, _, _ = clipped_env.step(action)

    assert abs(clipped_reward) < abs(reward)
    
    
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'MountainCar-v0'])
def test_clip_reward(env_id):
    env = gym.make(env_id)
    wrapped_env = ClipReward(env, -0.0005, 0.0002)

    env.reset()
    wrapped_env.reset()

    action = env.action_space.sample()

    _, reward, _, _ = env.step(action)
    _, wrapped_reward, _, _ = wrapped_env.step(action)

    assert abs(wrapped_reward) < abs(reward)
    assert wrapped_reward == -0.0005 or wrapped_reward == 0.0002
    

@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'MountainCar-v0', 
                                    'Pong-v0', 'SpaceInvaders-v0'])
def test_sign_clip_reward(env_id):
    env = gym.make(env_id)
    wrapped_env = SignClipReward(env)
    
    env.reset()
    wrapped_env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        _, wrapped_reward, done, _ = wrapped_env.step(action)
        assert wrapped_reward in [-1.0, 0.0, 1.0]
        if done:
            break
    
    
@pytest.mark.parametrize('env_id', ['Pong-v0', 'SpaceInvaders-v0'])
def test_flatten_observation(env_id):
    env = gym.make(env_id)
    wrapped_env = FlattenObservation(env)

    obs = env.reset()
    wrapped_obs = wrapped_env.reset()

    assert len(obs.shape) == 3
    assert len(wrapped_obs.shape) == 1
    assert wrapped_obs.shape[0] == obs.shape[0]*obs.shape[1]*obs.shape[2]
    
    
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
@pytest.mark.parametrize('num_stack', [2, 3, 4])
def test_frame_stack(env_id, num_stack):
    env = gym.make(env_id)
    shape = env.observation_space.shape
    env = FrameStack(env, num_stack)
    assert env.observation_space.shape == (num_stack,) + shape

    obs = env.reset()
    obs = np.asarray(obs)
    assert obs.shape == (num_stack,) + shape
    for i in range(1, num_stack):
        assert np.allclose(obs[i - 1], obs[i])

    obs, _, _, _ = env.step(env.action_space.sample())
    obs = np.asarray(obs)
    assert obs.shape == (num_stack,) + shape
    for i in range(1, num_stack - 1):
        assert np.allclose(obs[i - 1], obs[i])
    assert not np.allclose(obs[-1], obs[-2])
    
    
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
def test_get_wrapper(env_id):
    def make_env():
        return gym.make(env_id)

    env = make_env()
    env = ClipReward(env, 0.1, 0.5)
    env = FlattenObservation(env)
    env = FrameStack(env, 4)

    assert get_wrapper(env, 'ClipReward').__class__.__name__ == 'ClipReward'
    assert get_wrapper(env, 'FlattenObservation').__class__.__name__ == 'FlattenObservation'
    assert get_wrapper(env, 'Env') is None

    del env

    # vec_env
    env = make_vec_env(make_env, 3, 0)
    env = VecMonitor(env)
    assert get_wrapper(env, 'VecMonitor').__class__.__name__ == 'VecMonitor'
    assert get_wrapper(env, 'ClipReward') is None
    
    
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
def test_get_all_wrappers(env_id):
    def make_env():
        return gym.make(env_id)
    env = make_env()
    env = ClipReward(env, 0.1, 0.5)
    env = FlattenObservation(env)
    env = FrameStack(env, 4)
    assert get_all_wrappers(env) == ['FrameStack', 'FlattenObservation', 'ClipReward', 'TimeLimit']
    
    
@pytest.mark.parametrize('env_id', ['Pong-v0', 'SpaceInvaders-v0'])
@pytest.mark.parametrize('keep_dim', [True, False])
def test_gray_scale_observation(env_id, keep_dim):
    env = gym.make(env_id)
    wrapped_env = GrayScaleObservation(env, keep_dim=keep_dim)

    obs = env.reset()
    wrapped_obs = wrapped_env.reset()

    assert env.observation_space.shape[:2] == wrapped_env.observation_space.shape[:2]
    if keep_dim:
        assert wrapped_env.observation_space.shape[-1] == 1
        assert len(wrapped_obs.shape) == 3
    else:
        assert len(wrapped_env.observation_space.shape) == 2
        assert len(wrapped_obs.shape) == 2
    
    
@pytest.mark.parametrize('env_id', ['Pong-v0', 'SpaceInvaders-v0'])
@pytest.mark.parametrize('size', [16, 32])
def test_resize_observation(env_id, size):
    env = gym.make(env_id)
    env = ResizeObservation(env, size)

    assert env.observation_space.shape[-1] == 3
    assert env.observation_space.shape[:2] == (size, size)
    obs = env.reset()
    assert obs.shape == (size, size, 3)
    
    
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
@pytest.mark.parametrize('scale', [0.1, 200])
def test_scale_reward(env_id, scale):
    env = gym.make(env_id)

    action = env.action_space.sample()

    env.seed(0)
    env.reset()
    _, reward, _, _ = env.step(action)
    
    wrapped_env = ScaleReward(env, scale)
    env.seed(0)
    wrapped_env.reset()
    _, wrapped_reward, _, _ = wrapped_env.step(action)

    assert wrapped_reward == scale*reward
    
    
@pytest.mark.parametrize('env_id', ['Pong-v0', 'SpaceInvaders-v0'])
def test_scaled_float_frame(env_id):
    env = gym.make(env_id)
    env = ScaledFloatFrame(env)
    assert np.allclose(env.observation_space.high, 1.0)
    assert np.allclose(env.observation_space.low, 0.0)
    obs = env.reset()
    assert np.alltrue(obs <= 1.0) and np.alltrue(obs >= 0.0)
    obs, _, _, _ = env.step(env.action_space.sample())
    assert np.alltrue(obs <= 1.0) and np.alltrue(obs >= 0.0)
    
    
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_time_aware_observation(env_id):
    env = gym.make(env_id)
    wrapped_env = TimeAwareObservation(env)

    assert wrapped_env.observation_space.shape[0] == env.observation_space.shape[0] + 1

    obs = env.reset()
    wrapped_obs = wrapped_env.reset()
    assert wrapped_env.t == 0.0
    assert wrapped_obs[-1] == 0.0
    assert wrapped_obs.shape[0] == obs.shape[0] + 1

    wrapped_obs, _, _, _ = wrapped_env.step(env.action_space.sample())
    assert wrapped_env.t == 1.0
    assert wrapped_obs[-1] == 1.0
    assert wrapped_obs.shape[0] == obs.shape[0] + 1

    wrapped_obs, _, _, _ = wrapped_env.step(env.action_space.sample())
    assert wrapped_env.t == 2.0
    assert wrapped_obs[-1] == 2.0
    assert wrapped_obs.shape[0] == obs.shape[0] + 1

    wrapped_obs = wrapped_env.reset()
    assert wrapped_env.t == 0.0
    assert wrapped_obs[-1] == 0.0
    assert wrapped_obs.shape[0] == obs.shape[0] + 1
    
    
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
@pytest.mark.parametrize('num_env', [1, 3, 5])
@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('mode', ['serial', 'parallel'])
def test_vec_monitor(env_id, num_env, init_seed, mode):
    make_env = lambda: gym.make(env_id)
    env = make_vec_env(make_env, num_env, init_seed, mode)
    env = VecMonitor(env)

    env.reset()
    counter = 0
    for _ in range(2000):
        actions = [env.action_space.sample() for _ in range(len(env))]
        _, _, dones, infos = env.step(actions)
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                assert 'last_observation' in info
                assert 'episode' in info
                assert 'return' in info['episode']
                assert 'horizon' in info['episode']
                assert 'time' in info['episode']
                assert env.episode_rewards[i] == 0.0
                assert env.episode_horizons[i] == 0.0
                counter += 1
    assert min(100, counter) == len(env.return_queue)
    assert min(100, counter) == len(env.horizon_queue)
