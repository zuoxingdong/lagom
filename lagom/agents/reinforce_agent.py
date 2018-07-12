import torch
from torch.distributions import Categorical

from .base_agent import BaseAgent
from lagom.core.transform import Standardize


class REINFORCEAgent(BaseAgent):
    """
    REINFORCE algorithm (no baseline)
    """
    def __init__(self, policy, optimizer, config, **kwargs):
        self.policy = policy
        self.optimizer = optimizer
        
        super().__init__(config, **kwargs)
        
    def choose_action(self, obs):
        # Convert to Tensor
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs).float()
            obs = obs.unsqueeze(0)  # make a batch dimension
            obs = obs.to(self.device)  # move to device
        
        out_policy = self.policy(obs)
                
        # Dictionary of output data
        output = {}
        output = {**out_policy}
                
        return output
        
    def learn(self, x):
        batch_policy_loss = []
        
        # Iterate over list of trajectories
        for trajectory in x:
            R = trajectory.all_discounted_returns
            if self.config['standardize_r']:  # encourage/discourage half of performed actions, i.e. [-1, 1]
                standardize = Standardize()
                R = standardize(R)
                
            # Calculate policy loss for this trajectory (all time steps)
            policy_loss = []
            for logprob, r in zip(trajectory.all_info(name='action_logprob'), R):
                policy_loss.append(-logprob*float(r))  # TODO: supports VecEnv
            policy_loss = torch.cat(policy_loss).sum()
            
            # Record the policy loss
            batch_policy_loss.append(policy_loss)
            
        # Compute total loss (average over trajectories)
        total_loss = torch.stack(batch_policy_loss).mean()  # use stack because each element is zero-dim tensor
        
        # Zero-out gradient buffer
        self.optimizer.zero_grad()
        # Backward pass and compute gradients
        total_loss.backward()
        # Update for one step
        self.optimizer.step()
        
        # Output dictionary for different losses
        # TODO: if no more backprop needed, record with .item(), save memory without store computation graph
        output = {}
        output['total_loss'] = total_loss
        output['batch_policy_loss'] = batch_policy_loss

        return output
            
        """
        # Iterate over batch of trajectories
        for epi_data in batch_data:
            R = epi_data['returns']
            if self.config['standardize_r']:  # encourage/discourage half of performed actions, i.e. [-1, 1]
                R = Standardize().process(R)
            
            # Calculate loss for the policy
            policy_loss = []
            for log_prob, r in zip(epi_data['logprob_actions'], R):
                policy_loss.append(-log_prob*r)
            
            # Batched loss for each episode
            batch_policy_loss.append(torch.cat(policy_loss).sum())
            
        # Compute total loss over the batch (average over batch, i.e. over episodes)
        total_loss = torch.cat(batch_policy_loss).mean()
        
        # Zero-out gradient buffer
        self.optimizer.zero_grad()
        # Backward pass and compute gradients
        total_loss.backward()
        # Update for one step
        self.optimizer.step()
        
        # Output dictionary for different losses
        output = {}
        output['total_loss'] = total_loss
        output['batch_policy_loss'] = batch_policy_loss

        return output
        """
    
    def save(self, filename):
        self.policy.network.save(filename)
    
    def load(self, filename):
        self.policy.network.load(filename)