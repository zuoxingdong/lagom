from lagom.envs import EnvSpec
from lagom.envs import make_gym_env

from lagom.envs import make_vec_env
from lagom.envs.vec_env import SerialVecEnv

from lagom.agents import RandomAgent


def test_random_agent():
    env = make_gym_env('Pendulum-v0', 0)
    env_spec = EnvSpec(env)
    agent = RandomAgent(None, env_spec)
    out = agent.choose_action(env.reset())
    assert isinstance(out, dict)
    assert 'action' in out and out['action'].shape == (1,)

    venv = make_vec_env(SerialVecEnv, make_gym_env, 'CartPole-v0', 3, 0)
    env_spec = EnvSpec(venv)
    agent = RandomAgent(None, env_spec)
    out = agent.choose_action(env.reset())
    assert isinstance(out, dict)
    assert 'action' in out and len(out['action']) == 3 and isinstance(out['action'][0], int)
