import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .base_agent import BaseAgent

from lagom.core.transform import Standardize


class REINFORCEAgent(BaseAgent):
    r"""REINFORCE agent (no baseline). """
    def __init__(self, config, device, policy, optimizer, **kwargs):
        self.policy = policy
        self.optimizer = optimizer
        
        super().__init__(config, device, **kwargs)
        
        # accumulated trained timesteps
        self.total_T = 0
        
    def choose_action(self, obs, info={}):
        if not torch.is_tensor(obs):
            obs = np.asarray(obs)
            assert obs.ndim >= 2, f'expected at least 2-dim for batched data, got {obs.ndim}'
            obs = torch.from_numpy(obs).float().to(self.device)
            
        if self.policy.recurrent and self.info['reset_rnn_states']:
            self.policy.reset_rnn_states(batch_size=obs.size(0))
            self.info['reset_rnn_states'] = False  # Done, turn off
            
        out_policy = self.policy(obs, 
                                 out_keys=['action', 'action_logprob', 
                                           'entropy', 'perplexity'], 
                                 info=info)
        
        return out_policy

    def learn(self, D, info={}):
        batch_policy_loss = []
        batch_entropy_loss = []
        batch_total_loss = []
        
        for trajectory in D:
            logprobs = trajectory.all_info('action_logprob')
            entropies = trajectory.all_info('entropy')
            Qs = trajectory.all_discounted_returns
            
            # Standardize: encourage/discourage half of performed actions
            if self.config['agent.standardize_Q']:
                Qs = Standardize()(Qs).tolist()
            
            # Estimate policy gradient for all time steps and record all losses
            policy_loss = []
            entropy_loss = []
            for logprob, entropy, Q in zip(logprobs, entropies, Qs):
                policy_loss.append(-logprob*Q)
                entropy_loss.append(-entropy)
                
            # Average losses over all time steps
            policy_loss = torch.stack(policy_loss).mean()
            entropy_loss = torch.stack(entropy_loss).mean()
            
            # Calculate total loss
            entropy_coef = self.config['agent.entropy_coef']
            total_loss = policy_loss + entropy_coef*entropy_loss
            
            # Record all losses
            batch_policy_loss.append(policy_loss)
            batch_entropy_loss.append(entropy_loss)
            batch_total_loss.append(total_loss)
            
        # Average loss over list of Trajectory
        policy_loss = torch.stack(batch_policy_loss).mean()
        entropy_loss = torch.stack(batch_entropy_loss).mean()
        loss = torch.stack(batch_total_loss).mean()
        
        # Train with estimated policy gradient
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config['agent.max_grad_norm'] is not None:
            clip_grad_norm_(parameters=self.policy.network.parameters(), 
                            max_norm=self.config['agent.max_grad_norm'], 
                            norm_type=2)
            
        if hasattr(self, 'lr_scheduler'):
            if 'train.iter' in self.config:  # iteration-based
                self.lr_scheduler.step()
            elif 'train.timestep' in self.config:  # timestep-based
                self.lr_scheduler.step(self.total_T)
            else:
                raise KeyError('expected `train.iter` or `train.timestep` in config, but got none of them')
                
        self.optimizer.step()
        
        # Accumulate trained timesteps
        self.total_T += sum([trajectory.T for trajectory in D])
        
        out = {}
        out['loss'] = loss.item()
        out['policy_loss'] = policy_loss.item()
        out['entropy_loss'] = entropy_loss.item()
        if hasattr(self, 'lr_scheduler'):
            out['current_lr'] = self.lr_scheduler.get_lr()

        return out
    
    def save(self, f):
        self.policy.network.save(f)
    
    def load(self, f):
        self.policy.network.load(f)
