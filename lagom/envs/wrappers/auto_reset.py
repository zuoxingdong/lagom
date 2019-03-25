import gym


class AutoReset(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            info['last_observation'] = observation
            observation = self.env.reset()
        return observation, reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
