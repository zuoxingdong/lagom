import numpy as np

from gym import RewardWrapper


class ClipReward(RewardWrapper):
    r""""Clip reward to [min, max]. """
    def __init__(self, env, min_r, max_r):
        super().__init__(env)
        self.min_r = min_r
        self.max_r = max_r
            
    def reward(self, reward):
        return np.clip(reward, self.min_r, self.max_r)


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'MountainCar-v0'])
def test_clip_reward(env_id):
    env = gym.make(env_id)
    wrapped_env = ClipReward(env, -0.0005, 0.0002)

    env.reset()
    wrapped_env.reset()

    action = env.action_space.sample()

    _, reward, _, _ = env.step(action)
    _, wrapped_reward, _, _ = wrapped_env.step(action)

    assert abs(wrapped_reward) < abs(reward)
    assert wrapped_reward == -0.0005 or wrapped_reward == 0.0002
