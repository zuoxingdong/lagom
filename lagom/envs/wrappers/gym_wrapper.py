from .wrapper import Wrapper

from lagom.envs.spaces import convert_gym_space


class GymWrapper(Wrapper):
    r"""Convert OpenAI Gym environment to lagom-compatible environment. 
    
    It also converts all spaces (e.g. observation and action) to lagom-compatible :class:`Space`.
    """
    def __init__(self, env):
        # Override this to avoid Env type sanity check
        self.env = env
        
    @property
    def observation_space(self):
        return convert_gym_space(self.env.observation_space)
    
    @property
    def action_space(self):
        return convert_gym_space(self.env.action_space)
    
    @property
    def T(self):
        return self.env.spec.max_episode_steps
    
    @property
    def max_episode_reward(self):
        return self.env.spec.reward_threshold
