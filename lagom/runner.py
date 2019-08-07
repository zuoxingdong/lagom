from abc import ABC
from abc import abstractmethod

from lagom.data import StepType
from lagom.data import TimeStep
from lagom.data import Trajectory
from lagom.envs.timestep_env import TimeStepEnv


class BaseRunner(ABC):
    r"""Base class for all runners.
    
    A runner is a data collection interface between the agent and the environment. 
        
    """ 
    @abstractmethod
    def __call__(self, agent, env, **kwargs):
        r"""Defines data collection via interactions between the agent and the environment.
        
        Args:
            agent (BaseAgent): agent
            env (Env): environment
            **kwargs: keyword arguments for more specifications. 
            
        """
        pass


class EpisodeRunner(BaseRunner):
    def __call__(self, agent, env, N, **kwargs):
        assert isinstance(env, TimeStepEnv)
        D = []
        for _ in range(N):
            traj = Trajectory()
            timestep = env.reset()
            traj.add(timestep, None)
            while not timestep.last():
                out_agent = agent.choose_action(timestep, **kwargs)
                action = out_agent.pop('raw_action')
                timestep = env.step(action)
                timestep.info = {**timestep.info, **out_agent}
                traj.add(timestep, action)
            D.append(traj)
        return D


class StepRunner(BaseRunner):
    def __init__(self, reset_on_call=True):
        self.reset_on_call = reset_on_call
        self.observation = None
        
    def __call__(self, agent, env, T, **kwargs):
        assert isinstance(env, TimeStepEnv)
        D = []
        traj = Trajectory()
        if self.reset_on_call or self.observation is None:
            timestep = env.reset()
        else:
            timestep = TimeStep(StepType.FIRST, observation=self.observation, reward=None, done=None, info=None)
        traj.add(timestep, None)
        for t in range(T):
            out_agent = agent.choose_action(timestep, **kwargs)
            action = out_agent.pop('raw_action')
            timestep = env.step(action)
            timestep.info = {**timestep.info, **out_agent}
            traj.add(timestep, action)
            if timestep.last():
                D.append(traj)
                traj = Trajectory()
                timestep = env.reset()
                traj.add(timestep, None)
        if traj.T > 0:
            D.append(traj)
        if not self.reset_on_call:
            self.observation = timestep.observation
        return D
