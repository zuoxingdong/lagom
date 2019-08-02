import numpy as np

import pytest

from lagom.utils import Seeder

import gym
from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import Tuple
from gym.spaces import Dict
from gym.wrappers import ClipReward

from lagom.envs import RecordEpisodeStatistics
from lagom.envs import NormalizeObservation
from lagom.envs import NormalizeReward
from lagom.envs import make_vec_env
from lagom.envs import VecEnv
from lagom.envs.wrappers import get_wrapper
from lagom.envs.wrappers import get_all_wrappers
from lagom.envs.wrappers import NormalizeAction
from lagom.envs.wrappers import FlattenObservation
from lagom.envs.wrappers import FrameStack
from lagom.envs.wrappers import GrayScaleObservation
from lagom.envs.wrappers import ScaleReward
from lagom.envs.wrappers import TimeAwareObservation
from lagom.envs.wrappers import VecMonitor
from lagom.envs.wrappers import VecStandardizeObservation
from lagom.envs.wrappers import VecStandardizeReward
from lagom.envs.wrappers import StepInfo
from lagom.envs.wrappers import VecStepInfo


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
        obs = env.reset()
        wrapped_obs = wrapped_env.reset()
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

    
def test_normalize_action():
    env = gym.make('CartPole-v1')
    with pytest.raises(AssertionError):
        env = NormalizeAction(env)
    del env
    
    env = gym.make('Pendulum-v0')
    env = NormalizeAction(env)
    env.reset()
    with pytest.raises(AssertionError):
        env.step(10+env.action_space.sample())

    
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
def test_vec_monitor(env_id, num_env, init_seed):
    make_env = lambda: gym.make(env_id)
    env = make_vec_env(make_env, num_env, init_seed)
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

    
@pytest.mark.parametrize('num_env', [1, 3, 5])
@pytest.mark.parametrize('init_seed', [0, 10])
def test_vec_step_info(num_env, init_seed):
    make_env = lambda: gym.make('Pendulum-v0')
    env = make_vec_env(make_env, num_env, init_seed)
    env = VecStepInfo(env)

    observations, step_infos = env.reset()
    assert all([isinstance(step_info, StepInfo) for step_info in step_infos])
    assert all([step_info.first for step_info in step_infos])
    assert all([not step_info.mid for step_info in step_infos])
    assert all([not step_info.last for step_info in step_infos])
    assert all([not step_info.time_limit for step_info in step_infos])
    assert all([not step_info.terminal for step_info in step_infos])

    for _ in range(5000):
        observations, rewards, step_infos = env.step([env.action_space.sample() for _ in range(num_env)])
        for step_info in step_infos:
            assert isinstance(step_info, StepInfo)
            if step_info.last:
                assert step_info.done
                assert np.allclose(step_info['last_observation'], step_info.info['last_observation'])
                assert not step_info.first and not step_info.mid
                # Pendulum cut by TimeLimit
                assert 'TimeLimit.truncated' in step_info.info
                assert step_info.time_limit
                assert not step_info.terminal
            else:
                assert not step_info.done
                assert step_info.mid
                assert not step_info.first and not step_info.last
                assert not step_info.time_limit
                assert not step_info.terminal            
    del make_env, env

    make_env = lambda: gym.make('CartPole-v1')
    env = make_vec_env(make_env, num_env, init_seed)
    env = VecStepInfo(env)

    observations, step_infos = env.reset()
    assert all([isinstance(step_info, StepInfo) for step_info in step_infos])
    assert all([step_info.first for step_info in step_infos])
    assert all([not step_info.mid for step_info in step_infos])
    assert all([not step_info.last for step_info in step_infos])
    assert all([not step_info.time_limit for step_info in step_infos])
    assert all([not step_info.terminal for step_info in step_infos])

    for _ in range(5000):
        observations, rewards, step_infos = env.step([env.action_space.sample() for _ in range(num_env)])
        for step_info in step_infos:
            assert isinstance(step_info, StepInfo)
            if step_info.last:
                assert step_info.done
                assert np.allclose(step_info['last_observation'], step_info.info['last_observation'])
                assert not step_info.first and not step_info.mid
                # CartPole terminates episode with terminal state via random actions
                assert 'TimeLimit.truncated' not in step_info.info
                assert not step_info.time_limit
                assert step_info.terminal
            else:
                assert not step_info.done
                assert step_info.mid
                assert not step_info.first and not step_info.last
                assert not step_info.time_limit
                assert not step_info.terminal
