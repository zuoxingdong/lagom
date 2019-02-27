from abc import ABC
from abc import abstractmethod


class BaseRunner(ABC):
    r"""Base class for all runners.
    
    A runner is a data collection interface between the agent and the environment. 
    For each calling of the runner, the agent will take actions and receive observation
    in and from an environment for a certain number of trajectories/segments and a certain
    number of time steps. 
    
    .. note::
        
        By default, the agent handles batched data returned from :class:`VecEnv` type of environment.
        
    """ 
    @abstractmethod
    def __call__(self, agent, env, T, **kwargs):
        r"""Run the agent in the environment for a number of time steps and collect all necessary interaction data. 
        
        Args:
            agent (BaseAgent): agent
            env (VecEnv): VecEnv type of environment
            T (int): number of time steps
            **kwargs: keyword arguments for more specifications. 
        """
        pass
