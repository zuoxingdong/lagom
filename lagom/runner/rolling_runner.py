import torch

from lagom.envs import EnvSpec

from lagom.history import History

from lagom.runner import BaseRunner


class RollingRunner(BaseRunner):
    def __init__(self, config, agent, env):
        super().__init__(config, agent, env)
        
        self.env_spec = EnvSpec(self.env)
        
        self.fresh_created = True  # flag to call reset for first time
        
    def __call__(self, T, reset=False):
        D = History(self.env_spec, T)
        
        if self.fresh_created or reset:
            D.observations[:, 0, ...] = self.env.reset()
            if self.fresh_created:
                self.fresh_created = False
            
            # reset agent: e.g. RNN states because initial observation
            self.agent.reset(self.config)
            
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
            
        return D
