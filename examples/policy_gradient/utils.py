import gym
from gym.wrappers import Monitor

from lagom.envs import GymEnv


def make_env(seed=None, monitor=False, monitor_dir=None):
    env = gym.make('CartPole-v1')
    if monitor:
        env = Monitor(env, directory=monitor_dir)
    env = GymEnv(env)
    
    if seed is not None:
        env.seed(seed)
    
    return env
