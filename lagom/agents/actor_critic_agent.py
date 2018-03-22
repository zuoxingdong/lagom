import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from lagom.agents.base import BaseAgent
from lagom.core.processor import Standardize

class ActorCriticAgent(BaseAgent):
    """
    Actor-Critic with value network
    """
    def __init__(self, policy, optimizer, config):
        self.policy = policy
        self.optimizer = optimizer
        
        super().__init__(config)
        
    def choose_action(self, obs):
        out_policy = self.policy(obs)
        # Unpack output from policy network
        action_probs = out_policy.get('action_probs', None)
        state_value = out_policy.get('state_value', None)
        
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
        output['state_value'] = state_value

        return output
        
    def learn(self, batch_data):
        batch_policy_loss = []
        batch_value_loss = []

        # Iterate over batch of trajectories
        for epi_data in batch_data:
            R = epi_data['returns']
            if self.config['standardize_r']:  # encourage/discourage half of performed actions, i.e. [-1, 1]
                R = Standardize().process(R)

            # Calculate loss for the policy
            policy_loss = []
            value_loss = []
            for log_prob, value, r in zip(epi_data['logprob_actions'], epi_data['state_values'], R):
                advantage_estimate = r - value.data[0][0]
                policy_loss.append(-log_prob*advantage_estimate)
                value_loss.append(F.smooth_l1_loss(value, Variable(torch.FloatTensor([r]))))

            # Batched loss for each episode
            batch_policy_loss.append(torch.cat(policy_loss).sum())
            batch_value_loss.append(torch.cat(value_loss).sum())

        # Compute total loss over the batch (average over batch)
        total_loss = torch.mean(torch.cat(batch_policy_loss) + torch.cat(batch_value_loss))

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
        output['batch_value_loss'] = batch_value_loss

        return output