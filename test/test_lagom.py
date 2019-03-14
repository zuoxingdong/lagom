import pytest

import numpy as np

import gym

from pathlib import Path

from lagom.envs import make_vec_env
from lagom import RandomAgent
from lagom import Logger
from lagom.utils import pickle_load

    
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
