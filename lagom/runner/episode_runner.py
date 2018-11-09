import torch
import numpy as np

from lagom.envs import EnvSpec

from lagom.history import History

from lagom.runner import BaseRunner


class EpisodeRunner(BaseRunner):
    def __init__(self, config, agent, env):
        super().__init__(config, agent, env)
        
        self.env_spec = EnvSpec(self.env)
        
    def __call__(self, T):
        D = History(self.env_spec, T)
        
        D.observations[:, 0, ...] = self.env.reset()
        # reset agent: e.g. RNN states because initial observation
        self.agent.reset(self.config)
        
        track_done = [False]*D.N
        
        for t in range(T):
            info = {}
            out_agent = self.agent.choose_action(D.observations[:, 0, ...], info=info)
            
            action = out_agent.pop('action')
            if torch.is_tensor(action):
                raw_action = list(action.detach().cpu().numpy())
            else:
                raw_action = action
            D.add_t('actions', t, action)
            
            obs_next, reward, done, info = self.env.step(raw_action)
            D.observations[:, t+1, ...] = obs_next
            D.rewards[:, t] = reward
            D.dones[:, t] = done
            D.infos[t] = info
            
            # Record other information: e.g. log-probability of action, policy entropy
            [D.add_t(key, t, val) for key, val in out_agent.items()]
            
            track_done = np.logical_or(track_done, done)
            if np.all(track_done):  # early termination if all episodes completed
                # shrink size correspondingly
                D.T = t + 1
                D.observations = D.observations[:, :D.T+1, ...]
                D.rewards = D.rewards[:, :D.T]
                D.dones = D.dones[:, :D.T]
                D.infos = D.infos[:D.T]
                break
                
        for i, done in enumerate(track_done):
            if done:
                first_done_idx = np.where(D.dones[i] == True)[0][0]
                D.observations[i, first_done_idx+1:, ...] = 0.0  # eliminate init_observation
                D.rewards[i, first_done_idx+1:] = 0.0  # keep reward when done=True
                D.dones[i, first_done_idx:] = True
        
        return D
