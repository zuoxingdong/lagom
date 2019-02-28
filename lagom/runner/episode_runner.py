import numpy as np

from lagom.envs import VecEnv

from .base_runner import BaseRunner
from .batch_history import BatchHistory


class EpisodeRunner(BaseRunner):
    def __call__(self, agent, env, T, **kwargs):
        assert isinstance(env, VecEnv)
        D = BatchHistory(env)
        
        dones = [False for _ in range(env.num_env)]  # for RNN
        
        observations = env.reset()
        if agent.is_recurrent:
            agent.reset()  # e.g. RNN initial states
        for t in range(T):
            out_agent = agent.choose_action(observations, dones=dones, **kwargs)
            actions = out_agent.pop('raw_action')
            next_observations, rewards, dones, infos = env.step(actions)
            D.add(observations, actions, rewards, dones, infos, out_agent)
            observations = next_observations
            
            [D.stop(n) for n, done in enumerate(dones) if done]
            if all(D.stops):
                break
        for n, done in enumerate(dones):
            if not D.stops[n] and not done:
                D.info[n][-1][-1]['last_observation'] = observations[n]
                
        return D
