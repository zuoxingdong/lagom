import torch
from torch.autograd import Variable
from torch.distributions import Categorical

from utils import normalize_vec

class REINFORCEAgent(object):
    def __init__(self, policy, optimizer):
        self.policy = policy
        self.optimizer = optimizer
        
    def choose_action(self, state):
        # Convert state into Variable and FloatTensor with batch dimension
        state = Variable(torch.FloatTensor(state)).unsqueeze(0)
        # Compute probability distribution over actions via softmax
        action_probs = self.policy(state)
        # Sample an action according to categorical distribution
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        # log-probability of action
        logprob_action = action_dist.log_prob(action)
        
        # Convert action from Variable to scalar value
        action = action.data[0]
                
        # Dictionary of output data
        output = {}
        output['action'] = action
        output['log_prob'] = logprob_action
                
        return output
        
    def train(self, data_batch, normalize_r=False):
        """
        Update the agent for one step according to optimizer for collected batch of data 
        
        Args:
            data_batch: Collected batch of data outputs from Runner
            normalize_r: If True, then normalize the rewards
            
        Returns:
            batch_policy_loss: Loss for the policy network in the data batch
        """
        batch_policy_loss = []
        
        # Iterate over batch of trajectories
        for epi_data in data_batch:
            if normalize_r:
                R = normalize_vec(epi_data['returns'])
            else:
                R = epi_data['returns']
            
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