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
        out_policy = self.policy(obs, out_keys=['action', 'action_logprob', 'state_value'
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
            
            
            
            
            
            
                
            # Get all state values (without V_s_next in final transition)
            Vs = trajectory.all_info('V_s')
            
            # Advantage estimates
            As = [Q - V.item() for Q, V in zip(Qs, Vs)]
            
            # Get all log-probabilities and entropies
            logprobs = trajectory.all_info('action_logprob')
            entropies = trajectory.all_info('entropy')
            
            # Estimate policy gradient for all time steps and record all losses
            policy_loss = []
            value_loss = []
            entropy_loss = []
            for logprob, entropy, A, Q, V in zip(logprobs, entropies, As, Qs, Vs):
                policy_loss.append(-logprob*A)
                value_loss.append(F.mse_loss(V, torch.tensor(Q).view_as(V).to(V.device)))
                entropy_loss.append(-entropy)
                
            # Average over losses for all time steps
            policy_loss = torch.stack(policy_loss).mean()
            value_loss = torch.stack(value_loss).mean()
            entropy_loss = torch.stack(entropy_loss).mean()
        
            # Calculate total loss
            value_coef = self.config['agent:value_coef']
            entropy_coef = self.config['agent:entropy_coef']
            total_loss = policy_loss + value_coef*value_loss + entropy_coef*entropy_loss
            
            # Record all losses
            batch_policy_loss.append(policy_loss)
            batch_value_loss.append(value_loss)
            batch_entropy_loss.append(entropy_loss)
            batch_total_loss.append(total_loss)
        
        # Compute loss (average over trajectories)
        loss = torch.stack(batch_total_loss).mean()  # use stack because each element is zero-dim tensor
        
        # Zero-out gradient buffer
        self.optimizer.zero_grad()
        # Backward pass and compute gradients
        loss.backward()
        
        # Clip gradient norms if required
        if self.config['agent:max_grad_norm'] is not None:
            nn.utils.clip_grad_norm_(parameters=self.policy.network.parameters(), 
                                     max_norm=self.config['agent:max_grad_norm'], 
                                     norm_type=2)
        
        # Decay learning rate if required
        if hasattr(self, 'lr_scheduler'):
            if 'train:iter' in self.config:  # iteration-based training, so just increment epoch by default
                self.lr_scheduler.step()
            elif 'train:timestep' in self.config:  # timestep-based training, increment with timesteps
                self.lr_scheduler.step(self.accumulated_trained_timesteps)
            else:
                raise KeyError('expected train:iter or train:timestep in config, but none of them exist')
        
        # Take a gradient step
        self.optimizer.step()
        
        # Accumulate trained timesteps
        self.accumulated_trained_timesteps += sum([trajectory.T for trajectory in D])
        
        # Output dictionary for different losses
        # TODO: if no more backprop needed, record with .item(), save memory without store computation graph
        output = {}
        output['loss'] = loss
        output['batch_policy_loss'] = batch_policy_loss
        output['batch_value_loss'] = batch_value_loss
        output['batch_entropy_loss'] = batch_entropy_loss
        output['batch_total_loss'] = batch_total_loss
        if hasattr(self, 'lr_scheduler'):
            output['current_lr'] = self.lr_scheduler.get_lr()
        
        return output
    
    def save(self, filename):
        self.policy.network.save(filename)
    
    def load(self, filename):
        self.policy.network.load(filename)
