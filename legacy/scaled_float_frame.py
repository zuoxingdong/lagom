import numpy as np

from gym.spaces import Box
from gym import ObservationWrapper


class ScaledFloatFrame(ObservationWrapper):
    r"""Convert image frame to float range [0, 1] by dividing 255. 
    
    .. warning::
    
        Do NOT use this wrapper for DQN ! It will break the memory optimization.
    
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=1, shape=self.observation_space.shape, dtype=np.float32)
    
    def observation(self, observation):
        return observation.astype(np.float32)/255.


@pytest.mark.parametrize('env_id', ['Pong-v0', 'SpaceInvaders-v0'])
def test_scaled_float_frame(env_id):
    env = gym.make(env_id)
    env = ScaledFloatFrame(env)
    assert np.allclose(env.observation_space.high, 1.0)
    assert np.allclose(env.observation_space.low, 0.0)
    obs = env.reset()
    assert np.alltrue(obs <= 1.0) and np.alltrue(obs >= 0.0)
    obs, _, _, _ = env.step(env.action_space.sample())
    assert np.alltrue(obs <= 1.0) and np.alltrue(obs >= 0.0)
