import numpy as np

from gym import ActionWrapper


class ClipAction(ActionWrapper):
    r"""Clip the continuous action within the valid bound. """
    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)
