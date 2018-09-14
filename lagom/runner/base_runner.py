from lagom.envs.vec_env import VecEnv


class BaseRunner(object):
    r"""Base class for all runners.
    
    Any runner should subclass this class. 
    
    A runner is a data collection interface between the agent and the environment. 
    For each calling of the runner, the agent will take actions and receive observation
    in and from an environment for a certain number of trajectories/segments and a certain
    number of time steps. 
    
    .. note::
        
        By default, the agent handles batched data returned from :class:`VecEnv` type of environment.
        And the collected data should use either :class:`Trajectory` or :class:`Segment`. 
        
    """
    def __init__(self, agent, env, gamma):
        r"""Initialize the runner. 
        
        Args:
            agent (BaseAgent): agent
            env (VecEnv): VecEnv type of environment
            gamma (float): discount factor
        """
        assert isinstance(env, VecEnv), f'expected VecEnv, got {type(env)}'
        
        self.agent = agent
        self.env = env
        self.gamma = gamma
        
    def __call__(self, N, T):
        r"""Run the agent in the environment and collect all necessary interaction data as a batch. """
        raise NotImplementedError
