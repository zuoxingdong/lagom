import gym

from lagom.data import StepType
from lagom.data import TimeStep


class TimeStepEnv(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        step_type = StepType.LAST if done else StepType.MID
        timestep = TimeStep(step_type=step_type, observation=observation, reward=reward, done=done, info=info)
        return timestep

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return TimeStep(StepType.FIRST, observation=observation, reward=None, done=None, info=None)
