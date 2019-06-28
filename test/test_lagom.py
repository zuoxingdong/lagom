from pathlib import Path
import pytest
import numpy as np

import gym
from gym.wrappers import TimeLimit

from lagom import RandomAgent
from lagom import Logger
from lagom import EpisodeRunner
from lagom.metric import Trajectory
from lagom.envs import make_vec_env
from lagom.envs.wrappers import VecStepInfo
from lagom.utils import pickle_load

from .sanity_env import SanityEnv

    
def test_logger():
    logger = Logger()

    logger('iteration', 1)
    logger('learning_rate', 1e-3)
    logger('train_loss', 0.12)
    logger('eval_loss', 0.14)

    logger('iteration', 2)
    logger('learning_rate', 5e-4)
    logger('train_loss', 0.11)
    logger('eval_loss', 0.13)

    logger('iteration', 3)
    logger('learning_rate', 1e-4)
    logger('train_loss', 0.09)
    logger('eval_loss', 0.10)

    def check(logs):
        assert len(logs) == 4
        assert list(logs.keys()) == ['iteration', 'learning_rate', 'train_loss', 'eval_loss']
        assert logs['iteration'] == [1, 2, 3]
        assert np.allclose(logs['learning_rate'], [1e-3, 5e-4, 1e-4])
        assert np.allclose(logs['train_loss'], [0.12, 0.11, 0.09])
        assert np.allclose(logs['eval_loss'], [0.14, 0.13, 0.10])

    check(logger.logs)

    logger.dump()
    logger.dump(border='-'*50)
    logger.dump(keys=['iteration'])
    logger.dump(keys=['iteration', 'train_loss'])
    logger.dump(index=0)
    logger.dump(index=[1, 2])
    logger.dump(index=0)
    logger.dump(keys=['iteration', 'eval_loss'], index=1)
    logger.dump(keys=['iteration', 'learning_rate'], indent=1)
    logger.dump(keys=['iteration', 'train_loss'], index=[0, 2], indent=1, border='#'*50)

    f = Path('./logger_file')
    logger.save(f)
    f = f.with_suffix('.pkl')
    assert f.exists()

    logs = pickle_load(f)
    check(logs)

    f.unlink()
    assert not f.exists()

    logger.clear()
    assert len(logger.logs) == 0

    
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
@pytest.mark.parametrize('num_env', [1, 5])
def test_random_agent(env_id, num_env):
    make_env = lambda: gym.make(env_id)
    env = make_env()
    agent = RandomAgent(None, env, 'cpu')
    out = agent.choose_action(env.reset())
    assert isinstance(out, dict)
    assert out['raw_action'] in env.action_space
    del env, agent, out
    
    env = make_vec_env(make_env, num_env, 0)
    agent = RandomAgent(None, env, 'cpu')
    out = agent.choose_action(env.reset())
    assert isinstance(out, dict)
    assert len(out['raw_action']) == num_env
    assert all(action in env.action_space for action in out['raw_action'])


@pytest.mark.parametrize('env_id', ['Sanity', 'CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
@pytest.mark.parametrize('num_env', [1, 3])
@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('T', [1, 5, 100])
def test_episode_runner(env_id, num_env, init_seed, T):    
    if env_id == 'Sanity':
        make_env = lambda: TimeLimit(SanityEnv())
    else:
        make_env = lambda: gym.make(env_id)
    env = make_vec_env(make_env, num_env, init_seed)
    env = VecStepInfo(env)
    agent = RandomAgent(None, env, None)
    runner = EpisodeRunner()
    
    if num_env > 1:
        with pytest.raises(AssertionError):
            D = runner(agent, env, T)
    else:
        with pytest.raises(AssertionError):
            runner(agent, env.env, T)  # must be VecStepInfo
        D = runner(agent, env, T)
        for traj in D:
            assert isinstance(traj, Trajectory)
            assert len(traj) <= env.spec.max_episode_steps
            assert traj.numpy_observations.shape == (len(traj) + 1, *env.observation_space.shape)
            if isinstance(env.action_space, gym.spaces.Discrete):
                assert traj.numpy_actions.shape == (len(traj),)
            else:
                assert traj.numpy_actions.shape == (len(traj), *env.action_space.shape)
            assert traj.numpy_rewards.shape == (len(traj),)
            assert traj.numpy_dones.shape == (len(traj), )
            assert traj.numpy_masks.shape == (len(traj), )
            assert len(traj.step_infos) == len(traj)
            if traj.completed:
                assert np.allclose(traj.observations[-1], traj.step_infos[-1]['last_observation'])
