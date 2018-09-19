import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import BaseAgent

from lagom.core.transform import Standardize


class VPGAgent(BaseAgent):
    r"""Vanilla Policy Gradient (VPG) with value network (baseline), no bootstrapping to estimate value function. """
    def __init__(self, config, policy, optimizer, **kwargs):
        self.policy = policy
        self.optimizer = optimizer
        
        super().__init__(config, **kwargs)
        
        # accumulated trained timesteps
        self.total_T = 0
        
    def choose_action(self, obs):
        if not torch.is_tensor(obs):  # Tensor conversion, already batched observation
            obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)
            
        # Call policy: all metrics should be batched properly for Runner to work properly
        out_policy = self.policy(obs, out_keys=['action', 'action_logprob', 'state_value',
                                                'entropy', 'perplexity'])
        
        return out_policy
        
    def learn(self, D):
        out = {}
        
        batch_policy_loss = []
        batch_value_loss = []
        batch_entropy_loss = []
        batch_total_loss = []
        
        for trajectory in D:  # iterate over trajectories
            # Get all discounted returns as estimate of Q
            Qs = trajectory.all_discounted_returns
            
            # Standardize: encourage/discourage half of performed actions
            if self.config['agent.standardize_Q']:
                Qs = Standardize()(Qs).tolist()
            
            # Get all state values (not use `all_V` for tuple result as trajectory only has one episode data)
            Vs = trajectory.all_info('V_s')
            final_V = trajectory.transitions[-1].V_s_next
            final_done = trajectory.transitions[-1].done
            
            # Advantage estimates
            As = [Q - V.item() for Q, V in zip(Qs, Vs)]
            
            # Get all log-probabilities and entropies
            logprobs = trajectory.all_info('action_logprob')
            entropies = trajectory.all_info('entropy')
            
            # Estimate policy gradient for all time steps and record all losses
            policy_loss = []
            entropy_loss = []
            value_loss = []
            for logprob, entropy, A, Q, V in zip(logprobs, entropies, As, Qs, Vs):
                policy_loss.append(-logprob*A)
                entropy_loss.append(-entropy)
                value_loss.append(F.mse_loss(V, torch.tensor(Q).view_as(V).to(V.device)))
            if final_done:  # learn terminal state value as zero
                value_loss.append(F.mse_loss(final_V, torch.tensor(0.0).view_as(V).to(V.device)))    
            
            # Average over losses for all time steps
            policy_loss = torch.stack(policy_loss).mean()
            entropy_loss = torch.stack(entropy_loss).mean()
            value_loss = torch.stack(value_loss).mean()
            
            # Calculate total loss
            entropy_coef = self.config['agent.entropy_coef']
            value_coef = self.config['agent.value_coef']
            total_loss = policy_loss + value_coef*value_loss + entropy_coef*entropy_loss
            
            # Record all losses
            batch_policy_loss.append(policy_loss)
            batch_entropy_loss.append(entropy_loss)
            batch_value_loss.append(value_loss)
            batch_total_loss.append(total_loss)
                
        # Compute loss (average over trajectories): use `stack` as each is zero-dim
        loss = torch.stack(batch_total_loss).mean()
        policy_loss = torch.stack(batch_policy_loss).mean()
        entropy_loss = torch.stack(batch_entropy_loss).mean()
        value_loss = torch.stack(batch_value_loss).mean()        
            
        # Zero-out gradient buffer
        self.optimizer.zero_grad()
        # Backward pass and compute gradients
        loss.backward()
        
        # Clip gradient norms if required
        if self.config['agent.max_grad_norm'] is not None:
            nn.utils.clip_grad_norm_(parameters=self.policy.network.parameters(), 
                                     max_norm=self.config['agent.max_grad_norm'], 
                                     norm_type=2)
        
        # Decay learning rate if required
        if hasattr(self, 'lr_scheduler'):
            if 'train.iter' in self.config:  # iteration-based training, so just increment epoch by default
                self.lr_scheduler.step()
            elif 'train.timestep' in self.config:  # timestep-based training, increment with timesteps
                self.lr_scheduler.step(self.total_T)
            else:
                raise KeyError('expected `train.iter` or `train.timestep` in config, but none of them exist')
        
        # Take a gradient step
        self.optimizer.step()
        
        # Accumulate trained timesteps
        self.total_T += sum([trajectory.T for trajectory in D])
        
        # Output dictionary: use `item()` to save memory if no more backprop needed
        out['loss'] = loss.item()
        out['policy_loss'] = policy_loss.item()
        out['entropy_loss'] = entropy_loss.item()
        out['value_loss'] = value_loss.item()
        if hasattr(self, 'lr_scheduler'):
            out['current_lr'] = self.lr_scheduler.get_lr()

        return out
    
    def save(self, f):
        self.policy.network.save(f)
    
    def load(self, f):
        self.policy.network.load(f)
