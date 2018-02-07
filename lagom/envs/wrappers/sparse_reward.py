from lagom.envs.wrappers import RewardWrapper


class SparseReward(RewardWrapper):
    def process_reward(self, reward):
        if reward == 1:
            return reward
        else:
            return 0