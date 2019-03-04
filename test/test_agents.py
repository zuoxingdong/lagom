import numpy as np

import pytest

import gym

from lagom.envs import make_vec_env
from lagom.agents import RandomAgent


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
