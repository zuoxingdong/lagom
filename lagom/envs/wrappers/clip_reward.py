import numpy as np

from lagom.envs.wrappers import RewardWrapper


class ClipReward(RewardWrapper):
    r""""Clip reward to [-1, 1]. """
    def process_reward(self, reward):
        return np.clip(reward, -1, 1)
