import numpy as np

import pytest

from lagom.utils import Seeder

from lagom.envs import Env
from lagom.envs import EnvSpec

from lagom.envs.spaces import Box
from lagom.envs.spaces import Discrete
from lagom.envs.spaces import Dict
from lagom.envs.spaces import Tuple

import gym
from lagom.envs.spaces import convert_gym_space

from lagom.envs.wrappers import Wrapper
from lagom.envs.wrappers import ObservationWrapper
from lagom.envs.wrappers import GymWrapper
from lagom.envs.wrappers import FlattenObservation
from lagom.envs.wrappers import FrameStack
from lagom.envs.wrappers import RewardScale

from lagom.envs import make_gym_env
from lagom.envs import make_envs
from lagom.envs import make_vec_env

from lagom.envs.vec_env import VecEnv
from lagom.envs.vec_env import VecEnvWrapper
from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.vec_env import ParallelVecEnv
from lagom.envs.vec_env import VecStandardize
from lagom.envs.vec_env import VecClipAction
from lagom.envs.vec_env import get_wrapper


class TestSpaces(object):
    def test_box(self):
        with pytest.raises(AssertionError):
            Box(-1.0, 1.0, dtype=None)
        with pytest.raises(AssertionError):
            Box(-1.0, [1.0, 2.0], np.float32, shape=(2,))
        with pytest.raises(AssertionError):
            Box(np.array([-1.0, -2.0]), np.array([3.0, 4.0, 5.0]), np.float32)
        with pytest.raises(AttributeError):
            Box(np.array([-1.0, -2.0]), [3.0, 4.0], np.float32)

        def check(box):
            assert all([dtype == np.float32 for dtype in [box.dtype, box.low.dtype, box.high.dtype]])
            assert all([s == (2, 3) for s in [box.shape, box.low.shape, box.high.shape]])
            assert np.allclose(box.low, np.full([2, 3], -1.0))
            assert np.allclose(box.high, np.full([2, 3], 1.0))
            sample = box.sample()
            assert sample.shape == (2, 3) and sample.dtype == np.float32
            assert box.flat_dim == 6 and isinstance(box.flat_dim, int)
            assert box.flatten(sample).shape == (6,)
            assert np.allclose(sample, box.unflatten(box.flatten(sample)))
            assert sample in box
            assert str(box) == 'Box(2, 3)'
            assert box == Box(-1.0, 1.0, np.float32, shape=[2, 3])
            del box, sample

        box1 = Box(-1.0, 1.0, np.float32, shape=[2, 3])
        check(box1)

        x = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        box2 = Box(low=-x, high=x, dtype=np.float32)
        check(box2)

        assert box1 == box2

    def test_discrete(self):
        with pytest.raises(AssertionError):
            Discrete('no')
        with pytest.raises(AssertionError):
            Discrete(-1)

        discrete = Discrete(5)
        assert discrete.dtype == np.int32
        assert discrete.n == 5
        sample = discrete.sample()
        assert isinstance(sample, int)
        assert discrete.flat_dim == 5
        assert discrete.flatten(sample).shape == (5,)
        assert sample == discrete.unflatten(discrete.flatten(sample))
        assert sample in discrete
        assert discrete == Discrete(5)
        
    def test_dict(self):
        with pytest.raises(AssertionError):
            Dict([Discrete(10), Box(-1, 1, np.float32, shape=(3,))])

        sensor_space = Dict({'position': Box(-100, 100, np.float32, shape=(3,)), 
                             'velocity': Box(-1, 1, np.float32, shape=(3,))})
        assert len(sensor_space.spaces) == 2
        assert 'position' in sensor_space.spaces and 'velocity' in sensor_space.spaces
        assert sensor_space.spaces['position'] == Box(-100, 100, np.float32, shape=(3,))
        assert sensor_space.spaces['velocity'] == Box(-1, 1, np.float32, shape=(3,))
        space = Dict({'sensors': sensor_space, 'score': Discrete(100)})
        assert len(space.spaces) == 2
        assert 'sensors' in space.spaces and 'score' in space.spaces
        assert space.spaces['sensors'] == sensor_space
        assert space.spaces['score'] == Discrete(100)
        sample = space.sample()
        assert isinstance(sample, dict) and len(sample) == 2
        assert isinstance(sample['sensors'], dict) and len(sample['sensors']) == 2
        assert sample['sensors'] in sensor_space
        assert sample['score'] in Discrete(100)
        assert space.flat_dim == 3+3+100
        assert space.flatten(sample).shape == (106,)
        sample2 = space.unflatten(space.flatten(sample))
        assert sample['score'] == sample2['score']
        assert np.allclose(sample['sensors']['position'], sample2['sensors']['position'])
        assert np.allclose(sample['sensors']['velocity'], sample2['sensors']['velocity'])
        assert sample in space
        
    def test_tuple(self):
        with pytest.raises(AssertionError):
            Tuple(Discrete(10))

        space = Tuple([Discrete(5), 
                       Box(-1.0, 1.0, np.float32, shape=(2, 3)), 
                       Dict({'success': Discrete(2), 'velocity': Box(-1, 1, np.float32, shape=(1, 3))})])
        assert len(space.spaces) == 3
        assert space.spaces[0] == Discrete(5)
        assert space.spaces[1] == Box(-1.0, 1.0, np.float32, shape=(2, 3))
        assert space.spaces[2] == Dict({'success': Discrete(2), 'velocity': Box(-1, 1, np.float32, shape=(1, 3))})
        sample = space.sample()
        assert isinstance(sample, tuple) and len(sample) == 3
        assert sample[0] in Discrete(5)
        assert sample[1] in Box(-1.0, 1.0, np.float32, shape=(2, 3))
        assert sample[2] in Dict({'success': Discrete(2), 'velocity': Box(-1, 1, np.float32, shape=(1, 3))})
        assert space.flat_dim == 5+2*3+2+3
        assert space.flatten(sample).shape == (16,)
        sample2 = space.unflatten(space.flatten(sample))
        assert sample[0] == sample2[0]
        assert np.allclose(sample[1], sample2[1])
        assert sample[2]['success'] == sample2[2]['success']
        assert np.allclose(sample[2]['velocity'], sample2[2]['velocity'])
        assert sample in space
        
    def test_convert_gym_space(self):
        # Discrete
        gym_space = gym.spaces.Discrete(n=5)
        lagom_space = convert_gym_space(gym_space)
        assert isinstance(lagom_space, Discrete)
        assert not isinstance(lagom_space, gym.spaces.Discrete)
        assert lagom_space.n == 5
        assert lagom_space.sample() in lagom_space

        del gym_space, lagom_space

        # Box
        gym_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(2, 3), dtype=np.float32)
        lagom_space = convert_gym_space(gym_space)
        assert isinstance(lagom_space, Box)
        assert not isinstance(lagom_space, gym.spaces.Box)
        assert lagom_space.shape == (2, 3)
        assert lagom_space.sample() in lagom_space

        del gym_space, lagom_space

        # Dict
        gym_space = gym.spaces.Dict({
            'sensors': gym.spaces.Dict({
                'position': gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32), 
                'velocity': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)}), 
            'charge': gym.spaces.Discrete(100)})
        lagom_space = convert_gym_space(gym_space)
        assert isinstance(lagom_space, Dict)
        assert not isinstance(lagom_space, gym.spaces.Dict)
        assert len(lagom_space.spaces) == 2
        assert isinstance(lagom_space.spaces['sensors'], Dict)
        assert len(lagom_space.spaces['sensors'].spaces) == 2
        assert isinstance(lagom_space.spaces['charge'], Discrete)
        assert isinstance(lagom_space.spaces['sensors'].spaces['velocity'], Box)
        assert lagom_space.flat_dim == 100+3+3
        assert lagom_space.sample() in lagom_space

        del gym_space, lagom_space

        # Tuple
        gym_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Box(-1.0, 1.0, [2, 3], np.float32)))
        lagom_space = convert_gym_space(gym_space)
        assert isinstance(lagom_space, Tuple)
        assert not isinstance(lagom_space, gym.spaces.Tuple)
        assert len(lagom_space.spaces) == 2
        assert isinstance(lagom_space.spaces[0], Discrete)
        assert not isinstance(lagom_space.spaces[0], gym.spaces.Discrete)
        assert isinstance(lagom_space.spaces[1], Box)
        assert not isinstance(lagom_space.spaces[1], gym.spaces.Box)
        assert lagom_space.flat_dim == 2+2*3
        assert lagom_space.sample() in lagom_space

        del gym_space, lagom_space


# FlattenDictWrapper requires Mujoco, so omitted from test for CI
class TestWrappers(object):
    def test_gym_wrapper(self):
        gym_env = gym.make('CartPole-v1')
        env = GymWrapper(gym_env)
        assert isinstance(env, GymWrapper) and isinstance(env, Wrapper)
        assert isinstance(gym_env, gym.Env) and isinstance(env.unwrapped, gym.Env)
        assert isinstance(env.env, gym.Wrapper)
        assert env.reset().shape == (4,)
        assert isinstance(env.observation_space, Box)
        assert isinstance(env.action_space, Discrete)
        assert len(env.step(env.action_space.sample())) == 4
        assert env.seed(3) == [3]
        assert env.T == 500
        assert env.max_episode_reward == 475.0
        assert env.reward_range == (-float('inf'), float('inf'))

        del gym_env
        del env

        gym_env = gym.make('Pendulum-v0')
        env = GymWrapper(gym_env)
        assert isinstance(env, GymWrapper) and isinstance(env, Wrapper)
        assert isinstance(gym_env, gym.Env) and isinstance(env.unwrapped, gym.Env)
        assert isinstance(env.env, gym.Wrapper)
        assert env.reset().shape == (3,)
        assert isinstance(env.observation_space, Box)
        assert isinstance(env.action_space, Box)
        assert len(env.step(env.action_space.sample())) == 4
        assert env.seed(3) == [3]
        assert env.T == 200
        assert env.max_episode_reward is None
        assert env.reward_range == (-float('inf'), float('inf'))

        del gym_env
        del env
        
    def test_flatten_observation(self):
        gym_env = gym.make('Pong-v0')
        env = GymWrapper(gym_env)

        obs = env.reset()
        assert obs.shape == (210, 160, 3)

        env = FlattenObservation(env)
        obs = env.reset()
        assert obs.shape == (210*160*3,)
        
    def test_frame_stack(self):
        env = gym.make('CartPole-v1')
        env = GymWrapper(env)
        env = FrameStack(env, num_stack=4)
        env.seed(1)
        assert isinstance(env, Env) and isinstance(env, FrameStack)
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

    def test_reward_scale(self):
        env = gym.make('CartPole-v1')
        env = GymWrapper(env)
        env = RewardScale(env, scale=0.02)
        env.seed(1)
        env.reset()
        observation, reward, done, info = env.step(env.action_space.sample())
        assert reward == 0.02
        
        
class TestEnvs(object):
    def test_env_spec(self):
        env = gym.make('CartPole-v1')
        env = GymWrapper(env)
        env.seed(0)

        env_spec = EnvSpec(env)
        assert isinstance(env_spec.observation_space, Box)
        assert isinstance(env_spec.action_space, Discrete)
        assert env_spec.control_type == 'Discrete'
        assert env_spec.T == 500
        assert env_spec.max_episode_reward == 475.0
        assert env_spec.reward_range == (-float('inf'), float('inf'))
        
    def test_make_gym_env(self):
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
        assert not env_spec.is_vec_env

        with pytest.raises(TypeError):
            env_spec.num_env

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

    def test_make_envs(self):
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

    def test_make_vec_env(self):
        venv1 = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 5, 1)
        venv2 = make_vec_env(ParallelVecEnv, make_gym_env, 'CartPole-v1', 5, 1)
        assert isinstance(venv1, VecEnv) and isinstance(venv1, SerialVecEnv)
        assert isinstance(venv2, VecEnv) and isinstance(venv2, ParallelVecEnv)
        assert venv1.num_env == venv2.num_env
        env_spec1 = EnvSpec(venv1)
        assert env_spec1.num_env == venv1.num_env
        env_spec2 = EnvSpec(venv2)
        assert env_spec2.num_env == venv2.num_env
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

        
class TestVecEnv(object):
    @pytest.mark.parametrize('vec_env_class', [(0, SerialVecEnv), (1, ParallelVecEnv)])
    def test_vec_env(self, vec_env_class):
        # unpack class
        v_id, vec_env_class = vec_env_class

        venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1)
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
        assert env_spec.is_vec_env

        venv.close()
        assert venv.closed

    @pytest.mark.parametrize('vec_env_class', [SerialVecEnv, ParallelVecEnv])
    def test_vec_standardize(self, vec_env_class):
        venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1)
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
        venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1)
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
        venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1)
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
        venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1)
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
        venv = make_vec_env(vec_env_class, make_gym_env, 'CartPole-v1', 5, 1)
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

    def test_equivalence_vec_env(self):
        venv1 = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v1', 5, 1)
        venv2 = make_vec_env(ParallelVecEnv, make_gym_env, 'CartPole-v1', 5, 1)
        assert venv1.observation_space == venv2.observation_space
        assert venv1.action_space == venv2.action_space
        assert venv1.num_env == venv2.num_env
        obs1 = venv1.reset()
        obs2 = venv2.reset()
        assert np.allclose(obs1, obs2)
        a = [1]*5
        for _ in range(20):
            obs1, rewards1, dones1, _ = venv1.step(a)
            obs2, rewards2, dones2, _ = venv2.step(a)
            assert np.allclose(obs1, obs2)
            assert np.allclose(rewards1, rewards2)
            assert np.allclose(dones1, dones2)

            
def test_vec_clip_action():
    env = make_vec_env(SerialVecEnv, make_gym_env, 'MountainCarContinuous-v0', 2, 0)
    clipped_env = VecClipAction(env)
    
    action = [[0.5], [1000]]
    
    env.reset()
    _, rewards, _, _ = env.step(action)
    
    clipped_env.reset()
    _, rewards_clipped, _, _ = clipped_env.step(action)
    
    assert rewards[0] == rewards_clipped[0]
    assert abs(rewards[1]) > abs(rewards_clipped[1])


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_get_wrapper(env_id):
    env = make_vec_env(SerialVecEnv, make_gym_env, env_id, 3, 0)
    env = VecStandardize(env)
    env = VecClipAction(env)

    out = get_wrapper(env, 'VecClipAction')
    assert out.__class__.__name__ == 'VecClipAction'
    del out

    out = get_wrapper(env, 'VecStandardize')
    assert out.__class__.__name__ == 'VecStandardize'
    del out

    out = get_wrapper(env, 'SerialVecEnv')
    assert out.__class__.__name__ == 'SerialVecEnv'
