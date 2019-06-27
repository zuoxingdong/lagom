import numpy as np

from gym import ActionWrapper


class ClipAction(ActionWrapper):
    r"""Clip the continuous action within the valid bound. """
    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


def test_clip_action():
    # mountaincar: action-based rewards
    env = gym.make('MountainCarContinuous-v0')
    clipped_env = ClipAction(env)

    env.reset()
    clipped_env.reset()

    action = [10000.]

    _, reward, _, _ = env.step(action)
    _, clipped_reward, _, _ = clipped_env.step(action)

    assert abs(clipped_reward) < abs(reward)
