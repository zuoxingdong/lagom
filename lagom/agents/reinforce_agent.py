import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import BaseAgent

from lagom import Logger
from lagom.core.transform import Standardize


class REINFORCEAgent(BaseAgent):
    r"""REINFORCE agent (no baseline). """
    def __init__(self, config, policy, optimizer, **kwargs):
        self.policy = policy
        self.optimizer = optimizer
        
        super().__init__(config, **kwargs)
        
        self.accumulated_trained_timesteps = 0
        
    def choose_action(self, obs):
        if not torch.is_tensor(obs):  # Tensor conversion, already batched observation
            obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)
            
        # Call policy: all metrics should be batched properly for Runner to work properly
        out_policy = self.policy(obs)
        
        return out_policy

    def learn(self, D):
        out = {}
        
        logger = Logger()
        
        batch_policy_loss = []
        batch_entropy_loss = []
        batch_total_loss = []
        
        # Iterate over list of Trajectory in D
        for trajectory in D:
            # Get all discounted returns as estimate of Q
            Qs = trajectory.all_discounted_returns
            # TODO: when use GAE of TDs, really standardize it ? biased magnitude of learned value get wrong TD error
            # Standardize advantage estimates if required
            # encourage/discourage half of performed actions, respectively.
            if self.config['agent:standardize']:
                Qs = Standardize()(Qs)
            
            # Get all log-probabilities and entropies
            logprobs = trajectory.all_info('action_logprob')
            entropies = trajectory.all_info('entropy')
            
            # Estimate policy gradient for all time steps and record all losses
            policy_loss = []
            entropy_loss = []
            for logprob, entropy, Q in zip(logprobs, entropies, Qs):
                policy_loss.append(-logprob*Q)
                entropy_loss.append(-entropy)
                
            # Average over losses for all time steps
            policy_loss = torch.stack(policy_loss).mean()
            entropy_loss = torch.stack(entropy_loss).mean()
            
            # Calculate total loss
            entropy_coef = self.config['agent:entropy_coef']
            total_loss = policy_loss + entropy_coef*entropy_loss
            
            # Record all losses
            batch_policy_loss.append(policy_loss)
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
        output['batch_entropy_loss'] = batch_entropy_loss
        output['batch_total_loss'] = batch_total_loss
        if hasattr(self, 'lr_scheduler'):
            output['current_lr'] = self.lr_scheduler.get_lr()

        return output
    
    def save(self, f):
        self.policy.network.save(f)
    
    def load(self, f):
        self.policy.network.load(f)
