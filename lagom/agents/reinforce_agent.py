import torch
from torch.distributions import Categorical

from lagom.agents.base import BaseAgent
from lagom.core.processor import Standardize

class REINFORCEAgent(BaseAgent):
    def __init__(self, policy, optimizer):
        super().__init__(policy, optimizer)
        
    def choose_action(self, x):
        """
        Action selection according to internal policies
        
        Args:
            x: A dictionary with keys of different kind of data. 
                    Possible keys: ['observation', 'current_state', 'goal_state']
            
        Returns:
            output: A dictionary with keys of different kind of data.
                    Possible keys: ['action', 'log_prob']
        """
        out_policy = self.policy(x)
        
        # Unpack output from policy network
        action_probs = out_policy.get('action_probs', None)
        state_value = out_policy.get('state_value', None)
        
        # Sample an action according to categorical distribution
        action_dist = Categorical(action_probs)
        # Sample an action and calculate its log-probability according to distribution
        action = action_dist.sample()
        logprob_action = action_dist.log_prob(action)
        
        # Convert action from Variable to scalar value
        action = action.data[0]
                
        # Dictionary of output data
        output = {}
        output['action'] = action
        output['log_prob'] = logprob_action
                
        return output
        
    def learn(self, data_batch, standardize_r=False):
        """
        Update the agent for one step according to optimizer for collected batch of data 
        
        Args:
            data_batch: Collected batch of data outputs from Runner
            standardize_r: If True, then standardize the rewards
            
        Returns:
            batch_policy_loss: Loss for the policy network in the data batch
        """
        batch_policy_loss = []
        
        # Iterate over batch of trajectories
        for epi_data in data_batch:
            R = epi_data['returns']
            if standardize_r:  # encourage/discourage half of performed actions
                R = Standardize().process(R)
            
            # Calculate loss for the policy
            policy_loss = []
            for log_prob, r in zip(epi_data['logprob_actions'], R):
                policy_loss.append(-log_prob*r)
                
            # Batched loss for each episode
            batch_policy_loss.append(torch.cat(policy_loss).sum())
            
        # Compute total loss over the batch (average over batch)
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