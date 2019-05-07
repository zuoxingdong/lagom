from lagom.envs import VecEnv
from lagom.envs.wrappers import VecStepInfo

from .base_runner import BaseRunner
from .trajectory import Trajectory


class EpisodeRunner(BaseRunner):
    def __init__(self, reset_on_call=True):
        self.reset_on_call = reset_on_call
        self.observation = None
    
    def __call__(self, agent, env, T, **kwargs):
        assert isinstance(env, VecEnv)
        assert isinstance(env, VecStepInfo)
        assert len(env) == 1, 'for cleaner API, one should use single VecEnv'
        
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
            next_observation, reward, step_info = env.step(action)
            # unbatch for [reward, step_info]
            reward, step_info = map(lambda x: x[0], [reward, step_info])
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
