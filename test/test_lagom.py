from pathlib import Path
import pytest
import numpy as np

import gym

from lagom import Logger
from lagom import RandomAgent
from lagom import StepType
from lagom import TimeStep
from lagom import Trajectory
from lagom import EpisodeRunner
from lagom import StepRunner
from lagom.envs import TimeStepEnv
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

    
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
@pytest.mark.parametrize('num_envs', [1, 5])
def test_random_agent(env_id, num_envs):
    # vanilla environment
    env = gym.make(env_id)
    agent = RandomAgent(None, env, 'cpu')
    out = agent.choose_action(env.reset())
    assert isinstance(out, dict)
    assert out['raw_action'] in env.action_space
    del env, agent, out
    
    # vectorized environment
    env = gym.vector.make(env_id, num_envs)
    agent = RandomAgent(None, env, 'cpu')
    out = agent.choose_action(env.reset())
    assert isinstance(out, dict)
    assert isinstance(out['raw_action'], list)
    assert len(out['raw_action']) == env.num_envs
    assert all([action in env.action_space for action in out['raw_action']])


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_timestep(env_id):
    env = gym.make(env_id)
    observation = env.reset()
    timestep = TimeStep(StepType.FIRST, observation=observation, reward=None, done=None, info=None)
    assert timestep.first()
    assert not timestep.mid() and not timestep.last()
    assert not timestep.time_limit() and not timestep.terminal()

    for t in range(env.spec.max_episode_steps):
        observation, reward, done, info = env.step(env.action_space.sample())
        if done:
            timestep = TimeStep(StepType.LAST, observation=observation, reward=reward, done=done, info=info)
            assert timestep.last()
            assert not timestep.first() and not timestep.mid()
            if 'TimeLimit.truncated' in info:
                assert timestep.time_limit() and not timestep.terminal()
            else:
                assert timestep.terminal() and not timestep.time_limit()
            break
        else:
            timestep = TimeStep(StepType.MID, observation=observation, reward=reward, done=done, info=info)
            assert timestep.mid()
            assert not timestep.first() and not timestep.last()
            assert not timestep.time_limit() and not timestep.terminal()


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
def test_trajectory(env_id):
    env = gym.make(env_id)
    env = TimeStepEnv(env)
    traj = Trajectory()
    assert len(traj) == 0 and traj.T == 0

    with pytest.raises(AssertionError):
        traj.add(TimeStep(StepType.MID, 1, 2, True, {}), 0.5)

    timestep = env.reset()
    traj.add(timestep, None)
    assert len(traj) == 1 and traj.T == 0

    with pytest.raises(AssertionError):
        traj.add(TimeStep(StepType.MID, 1, 2, True, {}), None)

    while not timestep.last():
        action = env.action_space.sample()
        timestep = env.step(action)
        traj.add(timestep, action)
        if not timestep.last():
            assert len(traj) == traj.T + 1
            assert not traj.finished
    with pytest.raises(AssertionError):
        traj.add(timestep, 5.3)
    assert traj.finished
    assert traj.reach_time_limit == traj[-1].time_limit()
    assert traj.reach_terminal == traj[-1].terminal()
    assert np.asarray(traj.observations).shape == (traj.T + 1, *env.observation_space.shape)
    assert len(traj.actions) == traj.T
    assert len(traj.rewards) == traj.T
    assert len(traj.dones) == traj.T
    assert len(traj.infos) == traj.T
    if traj.reach_time_limit:
        assert len(traj.get_infos('TimeLimit.truncated')) == 1


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
@pytest.mark.parametrize('N', [1, 5, 10])
def test_episode_runner(env_id, N):
    env = gym.make(env_id)
    env = TimeStepEnv(env)
    agent = RandomAgent(None, env, None)
    runner = EpisodeRunner()
    D = runner(agent, env, N)
    assert len(D) == N
    assert all([isinstance(d, Trajectory) for d in D])
    assert all([traj.finished for traj in D])
    assert all([traj[0].first() for traj in D])
    assert all([traj[-1].last() for traj in D])
    for traj in D:
        for timestep in traj[1:-1]:
            assert timestep.mid()


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0'])
@pytest.mark.parametrize('T', [1, 100, 500])
def test_step_runner(env_id, T):
    env = gym.make(env_id)
    env = TimeStepEnv(env)
    agent = RandomAgent(None, env, None)

    runner = StepRunner(reset_on_call=True)
    D = runner(agent, env, T)
    assert runner.observation is None
    assert all([isinstance(traj, Trajectory) for traj in D])
    assert all([traj[0].first() for traj in D])
    assert all([traj[-1].last() for traj in D[:-1]])

    runner = StepRunner(reset_on_call=False)
    D = runner(agent, env, 1)
    assert D[0][0].first()
    assert len(D[0]) == 2
    assert np.allclose(D[0][-1].observation, runner.observation)
    D2 = runner(agent, env, 3)
    assert np.allclose(D2[-1][-1].observation, runner.observation)
    assert np.allclose(D[0][-1].observation, D2[0][0].observation)
    assert D2[0][0].first() and D2[0][0].reward is None
