import torch
from torch.distributions import Categorical

from lagom.agents.base import BaseAgent
from lagom.core.processor import Standardize

class REINFORCEAgent(BaseAgent):
    """
    REINFORCE algorithm (no baseline)
    """
    def __init__(self, policy, optimizer, config):
        self.policy = policy
        self.optimizer = optimizer
        
        super().__init__(config)
        
    def choose_action(self, obs):
        out_policy = self.policy(obs)
        # Unpack output from policy network
        action_probs = out_policy.get('action_probs', None)
        
        # Sample an action according to categorical distribution
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        # Calculate log-probability according to distribution
        logprob_action = action_dist.log_prob(action)
        
        # Convert action from Variable to scalar value
        action = action.data[0]
                
        # Dictionary of output data
        output = {}
        output['action'] = action
        output['logprob_action'] = logprob_action
                
        return output
        
    def learn(self, batch_data):
        batch_policy_loss = []
        
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