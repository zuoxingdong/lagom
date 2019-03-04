import numpy as np

from lagom.envs import VecEnv

from .base_runner import BaseRunner
from .batch_history import BatchHistory


class RollingSegmentRunner(BaseRunner):
    def __init__(self):
        self.obs_buffer = None  # for next call
        self.done_buffer = None  # masking
    
    def __call__(self, agent, env, T, reset=False, **kwargs):
        assert isinstance(env, VecEnv)
        D = BatchHistory(env)
    
        if self.obs_buffer is None or reset:
            observations = env.reset()
            dones = [False for _ in range(env.num_env)]  # for RNN
            if agent.is_recurrent:
                agent.reset()  # e.g. RNN initial states
        else:
            observations = self.obs_buffer
            dones = self.done_buffer
        for t in range(T):
            out_agent = agent.choose_action(observations, dones=dones, **kwargs)
            actions = out_agent.pop('raw_action')
            next_observations, rewards, dones, infos = env.step(actions)
            D.add(observations, actions, rewards, dones, infos, out_agent)
            observations = next_observations
        for n, done in enumerate(dones):
            if not done:
                D.info[n][-1][-1]['last_observation'] = observations[n]
        
        self.obs_buffer = observations
        self.done_buffer = dones
        
        return D
