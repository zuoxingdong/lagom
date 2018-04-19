from lagom.envs.base import Env
from lagom.envs.spaces import Dict


class GoalEnv(Env):
    """
    A goal-based environment. The observation space is a dictionary space consisting of 
    - 'observation': same like regular observation
    - 'achieved_goal': the goal is currently achieved by the agent
    - 'desired_goal': the goal should be achieved by the agent
    """
    def reset(self):
        # Ensure the observation space is Dictionary space
        if not isinstance(self.observation_space, Dict):
            raise TypeError('The observation space of GoalEnv must be of type Dict')
        observation = super().reset()
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key not in observation:
                raise KeyError(f'The key ({key}) must be contained in the observation dictionary. ')
        return observation
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute the step reward dependent on a desired goal and a goal that was achieved. 
        
        Args:
            achieved_goal (object): the goal that was achieved
            desired_goal (object): the desired goal that agent should achieve
            info (dict): additional information, e.g. additional rewards that are independent of goal. 
            
        Returns:
            reward (float): the reward according to achieved goal and desired goal.
                    Note that the following should be true:
                        observation, reward, done, info = env.step()
                        assert reward == env.compute_reward(observation['achieved_goal'], observation['desired_goal'], info)
        """
        raise NotImplementedError