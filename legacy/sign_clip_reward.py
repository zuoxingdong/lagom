import numpy as np

from gym import RewardWrapper


class SignClipReward(RewardWrapper):
    r""""Bin reward to {-1, 0, +1} by its sign. """   
    def reward(self, reward):
        return np.sign(reward)


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'MountainCar-v0', 
                                    'Pong-v0', 'SpaceInvaders-v0'])
def test_sign_clip_reward(env_id):
    env = gym.make(env_id)
    wrapped_env = SignClipReward(env)
    
    env.reset()
    wrapped_env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        _, wrapped_reward, done, _ = wrapped_env.step(action)
        assert wrapped_reward in [-1.0, 0.0, 1.0]
        if done:
            break
