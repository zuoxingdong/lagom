import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from lagom.agents.base import BaseAgent
from lagom.core.preprocessors import Standardize

class ActorCriticAgent(BaseAgent):
    """
    Actor-Critic with value network
    """
    def __init__(self, policy, optimizer, lr_scheduler, config):
        self.policy = policy
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        super().__init__(config)
        
    def choose_action(self, obs, mode):
        assert mode == 'sampling' or mode == 'greedy'
        
        out_policy = self.policy(obs)
        # Unpack output from policy network
        action_probs = out_policy['action_probs']
        state_value = out_policy['state_value']
        
        # Create a categorical distribution
        # TODO: automatic distribution select according to action space
        action_dist = Categorical(action_probs)
        # Calculate entropy of the policy conditional on state
        entropy = action_dist.entropy()
        # Calculate perplexity of the policy, i.e. exp(entropy)
        perplexity = action_dist.perplexity()
        
        if mode == 'greedy':  # greedily select an action, useful for evaluation
            action = torch.argmax(action_probs, 1)
            logprob_action = None  # due to greedy selection, no log-probability available
        elif mode == 'sampling':  # sample an action according to distribution
            action = action_dist.sample()
            logprob_action = action_dist.log_prob(action)  # calculate log-probability
            
            
            #print(f'#######{action_probs}')
            #print(f'!!!!!!!{action.item()}')
        
        # Dictionary of output data
        output = {}
        output['action'] = action
        output['logprob_action'] = logprob_action
        output['state_value'] = state_value
        output['entropy'] = entropy
        output['perplexity'] = perplexity

        return output
        
    def learn(self, batch):
        batch_policy_loss = []
        batch_value_loss = []
        batch_entropy_loss = []
        batch_total_loss = []

        for episode in batch:  # Iterate over batch of episodes
            # Get all returns
            Qs = episode.all_returns
            # Get all values
            Vs = episode.all_info('state_value')
            # Get all action log-probabilities
            log_probs = episode.all_info('logprob_action')
            # Get all entropies
            entropies = episode.all_info('entropy')
            
            # TODO: testing standardization before or after computing advantage estimation
            # Calculate advantage estimation
            Qs = Standardize().process(Qs)
            
            # Calculate losses
            policy_loss = []
            value_loss = []
            entropy_loss = []
            
            # iterate over time steps
            for logprob, V, Q in zip(log_probs, Vs, Qs):
                advantage_estimate = Q - V.item()
                policy_loss.append(-logprob*advantage_estimate)
                value_loss.append(F.mse_loss(V, torch.Tensor([Q])))
            
            
            
            
            
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