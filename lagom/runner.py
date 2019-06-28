from abc import ABC
from abc import abstractmethod

from lagom.metric import Trajectory
from lagom.envs import VecEnv
from lagom.envs.wrappers import VecStepInfo


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


class EpisodeRunner(BaseRunner):
    def __init__(self, reset_on_call=True):
        self.reset_on_call = reset_on_call
        self.observation = None
    
    def __call__(self, agent, env, T, **kwargs):
        assert isinstance(env, VecEnv) and isinstance(env, VecStepInfo) and len(env) == 1
        
        D = [Trajectory()]
        if self.reset_on_call:
            observation, _ = env.reset()
        else:
            if self.observation is None:
                self.observation, _ = env.reset()
            observation = self.observation
        D[-1].add_observation(observation)
        for t in range(T):
            out_agent = agent.choose_action(observation, **kwargs)
            action = out_agent.pop('raw_action')
            next_observation, [reward], [step_info] = env.step(action)
            step_info.info = {**step_info.info, **out_agent}
            if step_info.last:
                D[-1].add_observation([step_info['last_observation']])  # add a batch dim    
            else:
                D[-1].add_observation(next_observation)
            D[-1].add_action(action)
            D[-1].add_reward(reward)
            D[-1].add_step_info(step_info)
            if step_info.last:
                assert D[-1].completed
                D.append(Trajectory())
                D[-1].add_observation(next_observation)  # initial observation
            observation = next_observation
        if len(D[-1]) == 0:
            D = D[:-1]
        self.observation = observation
        return D
