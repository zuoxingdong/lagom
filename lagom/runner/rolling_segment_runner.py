import torch
import numpy as np

from lagom.envs import EnvSpec

from lagom.history import BatchSegment

from lagom.runner import BaseRunner


class RollingSegmentRunner(BaseRunner):
    def __init__(self, config, agent, env):
        super().__init__(config, agent, env)
        
        self.env_spec = EnvSpec(self.env)
        
        self.obs_buffer = None  # for next call
        self.done_buffer = None  # masking
        
    def __call__(self, T, reset=False):
        D = BatchSegment(self.env_spec, T)
        
        if self.obs_buffer is None or reset:
            obs = self.env.reset()
            # reset agent: e.g. RNN states because initial observation
            self.agent.reset(self.config)
        else:
            obs = self.obs_buffer
        D.add_observation(0, obs)
            
        for t in range(T):
            info = {}
            out_agent = self.agent.choose_action(obs, info=info)
            
            action = out_agent.pop('action')
            if torch.is_tensor(action):
                raw_action = list(action.detach().cpu().numpy())
            else:
                raw_action = action
            D.add_action(t, raw_action)
            
            obs, reward, done, info = self.env.step(raw_action)
            D.add_observation(t+1, obs)
            D.add_reward(t, reward)
            D.add_done(t, done)
            D.add_info(info)
        
            # Record other information: e.g. log-probability of action, policy entropy
            D.add_batch_info(out_agent)
        
        self.obs_buffer = obs
        self.done_buffer = done
        
        return D
