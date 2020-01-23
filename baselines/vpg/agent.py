import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym.spaces as spaces
import lagom
import lagom.utils as utils
import lagom.rl as rl


class MLP(lagom.nn.Module):
    def __init__(self, config, env, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        
        self.feature_layers = lagom.nn.make_fc(spaces.flatdim(env.observation_space), config['nn.sizes'])
        for layer in self.feature_layers:
            lagom.nn.ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in config['nn.sizes']])

    def forward(self, x):
        for layer, layer_norm in zip(self.feature_layers, self.layer_norms):
            x = F.gelu(layer_norm(layer(x)))
        return x


class Agent(lagom.BaseAgent):
    def __init__(self, config, env, **kwargs):
        super().__init__(config, env, **kwargs)
        
        feature_dim = config['nn.sizes'][-1]
        self.feature_network = MLP(config, env, **kwargs)
        if isinstance(env.action_space, spaces.Discrete):
            self.action_head = lagom.nn.CategoricalHead(feature_dim, env.action_space.n, **kwargs)
        elif isinstance(env.action_space, spaces.Box):
            self.action_head = lagom.nn.DiagGaussianHead(feature_dim, spaces.flatdim(env.action_space), config['agent.std0'], **kwargs)
        self.V_head = nn.Linear(feature_dim, 1)
        lagom.nn.ortho_init(self.V_head, weight_scale=1.0, constant_bias=0.0)

        self.optimizer = optim.Adam(self.parameters(), lr=config['agent.lr'])
        self.register_buffer('total_timestep', torch.tensor(0))
        
    def choose_action(self, x, **kwargs):
        obs = torch.as_tensor(x.observation).float().to(self.config.device).unsqueeze(0)
        features = self.feature_network(obs)
        action_dist = self.action_head(features)
        V = self.V_head(features)
        action = action_dist.sample()
        out = {}
        out['action_dist'] = action_dist
        out['V'] = V
        out['entropy'] = action_dist.entropy()
        out['action'] = action
        out['raw_action'] = utils.numpify(action.squeeze(0), self.env.action_space.dtype)
        out['action_logprob'] = action_dist.log_prob(action.detach())
        return out
    
    def learn(self, D, **kwargs):
        get_info = lambda key: [torch.cat(traj.get_infos(key)) for traj in D]
        logprobs, entropies, Vs = map(get_info, ['action_logprob', 'entropy', 'V'])
        last_Vs = [traj.extra_info['last_info']['V'] for traj in D]
        Qs = [rl.bootstrapped_returns(self.config['agent.gamma'], traj.rewards, last_V, traj.reach_terminal)
              for traj, last_V in zip(D, last_Vs)]
        As = [rl.gae(self.config['agent.gamma'], self.config['agent.gae_lambda'], traj.rewards, V, last_V, traj.reach_terminal)
              for traj, V, last_V in zip(D, Vs, last_Vs)]
        
        # Handle dtype, device
        logprobs, entropies, Vs = map(lambda x: torch.cat(x).squeeze(), [logprobs, entropies, Vs])
        Qs, As = map(lambda x: torch.as_tensor(np.concatenate(x)).float().squeeze().to(self.config.device), [Qs, As])
        if self.config['agent.standardize_adv']:
            As = (As - As.mean())/(As.std() + 1e-4)    
        assert all([x.ndim == 1 for x in [logprobs, entropies, Vs, Qs, As]])
        
        policy_loss = -logprobs*As.detach()
        entropy_loss = -entropies
        value_loss = F.mse_loss(Vs, Qs, reduction='none')
        loss = policy_loss + self.config['agent.value_coef']*value_loss + self.config['agent.entropy_coef']*entropy_loss
        loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.config['agent.max_grad_norm'])
        self.optimizer.step()
        self.total_timestep += sum([traj.T for traj in D])
        
        out = {}
        out['loss'] = loss.item()
        out['grad_norm'] = grad_norm
        out['policy_loss'] = policy_loss.mean().item()
        out['entropy_loss'] = entropy_loss.mean().item()
        out['policy_entropy'] = -out['entropy_loss']
        out['value_loss'] = value_loss.mean().item()
        out['V'] = utils.describe(Vs.squeeze(), axis=-1, repr_indent=1, repr_prefix='\n')
        out['explained_variance'] = utils.explained_variance(y_true=Qs.tolist(), y_pred=Vs.tolist())
        return out
    
    def checkpoint(self, logdir, num_iter):
        self.save(logdir/f'agent_{num_iter}.pth')
        if self.config['env.normalize_obs']:
            moments = (self.env.obs_moments.mean, self.env.obs_moments.var)
            utils.pickle_dump(obj=moments, f=logdir/f'obs_moments_{num_iter}', ext='.pth')
