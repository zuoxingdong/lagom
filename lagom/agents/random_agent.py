from .base_agent import BaseAgent

from lagom.envs import VecEnv


class RandomAgent(BaseAgent):
    r"""A random agent samples action uniformly from action space. """    
    def choose_action(self, obs, **kwargs):
        if isinstance(self.env, VecEnv):
            action = [self.env.action_space.sample() for _ in range(self.env.num_env)]
        else:
            action = self.env.action_space.sample()
        out = {'raw_action': action}
        return out  

    def learn(self, D, **kwargs):
        pass
