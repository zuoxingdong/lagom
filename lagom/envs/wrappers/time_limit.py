import gym

from .utils import get_all_wrappers


# TODO: temporary, remove after it is officially ported to gym
# So, no unit test in lagom
class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        assert self.__class__.__name__ not in get_all_wrappers(env), 'TimeLimit cannot be wrapped twice' 
        super().__init__(env)
        self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            if not done:
                info['TimeLimit.truncated'] = True
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
