import torch.nn as nn
import torch.nn.functional as F

from lagom.core.networks import MLP


class CategoricalMLPPolicy(MLP):
    """
    MLP policy network with categorical distribution (discrete actions)
    """
    def __init__(self, env_spec, config):
        """
        Args:
            env_spec (EnvSpec): Specifications of the environment.
            config (dict): A dictionary of configurations
        """
        self.env_spec = env_spec
        self.config = config
        
        # Make an MLP network
        super().__init__(input_dim=self.env_spec.observation_space.flat_dim,
                         hidden_sizes=self.config['hidden_sizes'], 
                         hidden_nonlinearity=self.config['hidden_nonlinearity'], 
                         output_dim=None,  # Separate action head
                         output_nonlinearity=None)
        
        # Action head
        in_features = self.hidden_sizes[-1]
        out_features = self.env_spec.action_space.flat_dim
        self.action_head = nn.Linear(in_features, out_features)
        
        # Value head
        if self.config['predict_value']:
            in_features = self.hidden_sizes[-1]
            out_features = 1
            self.value_head = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # Forward pass by internal MLP network
        x = super().forward(x)
        
        # Action head
        action_scores = self.action_head(x)
        action_probs = F.softmax(action_scores, dim=1)
        
        # Value head
        if self.config['predict_value']:
            state_value = self.value_head(x)
        
        # Output dictionary
        output = {}
        output['action_probs'] = action_probs
        if self.config['predict_value']:
            output['state_value'] = state_value
        
        return output