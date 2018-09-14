from .env import Env
from .spaces import Dict


class GoalEnv(Env):
    r"""A goal-based environment. 
    
    The observation space is a dictionary space (i.e. :class:`Dict`) with at least the following keys:
    
    * 'observation': same like usual observation
    * 'desired_goal': the goal should be achieved by the agent
    * 'achieved_goal': the goal is currently achieved by the agent
    
    The subclass should implement at least the following:

    - :meth:`step`
    - :meth:`compute_reward`
    - :meth:`render`
    - :meth:`close`
    - :meth:`seed`
    - :meth:`T`
    - :meth:`observation_space`
    - :meth:`action_space`
    - :meth:`max_episode_reward`
    - :meth:`reward_range`
    
    """
    def reset(self):
        # Ensure the observation space is Dictionary space
        assert isinstance(self.observation_space, Dict), f'expected Dict space, got {type(self.observation_space)}'
        # Get goal-based observation
        observation = super().reset()
        # Sanity check
        msg = 'must contain at least three keys, [observation, achieved_goal, desired_goal]'
        assert all([key in observation for key in ['observation', 'achieved_goal', 'desired_goal']]), msg
        
        return observation
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        r"""Compute the step reward depending on a desired goal and a goal that is currently achieved. 
        
        .. note::
        
            If it needs to include additional rewards that are independent of the goal, the values should
            be in the info and compute it accordingly. 
            
        .. note::
        
            The following should always hold true::
            
                import gym

                env = gym.make('FetchPush-v1')
                env.reset()
                observation, reward, done, info = env.step(env.action_space.sample())
                assert reward == env.compute_reward(observation['achieved_goal'], observation['desired_goal'], info)

        Args:
            achieved_goal (object): the goal that is currently achieved. 
            desired_goal (object): the desired goal that agent should achieve
            info (dict): a dictionary of additional information e.g. goal-independent rewards 
            
        Returns
        -------
        reward : float
            the reward based on currently achieved goal and desired goal.            
        """
        raise NotImplementedError
