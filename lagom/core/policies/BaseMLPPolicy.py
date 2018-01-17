import torch.nn as nn
import torch.nn.functional as F

#################################
# TODO: Continuous MLP policy (Categorical/Gaussian)

# utils.py function of make_policy(args.....)
#################################


class BaseMLPPolicy(nn.Module):
    """
    Base class for fully connected network (or Multi-Layer Perceptron)
    """
    def __init__(self, env_spec, fc_sizes=[128], predict_value=False):
        """
        Base class for fully connected network (or Multi-Layer Perceptron)
        
        Args:
            env_spec (EnvSpec): Specifications of the environment.
            fc_sizes: A list of number of hidden neurons for fully connected layer.
        """
        super().__init__()
        
        # Iteratively build network, should use nn.Sequential, otherwise cannot be recognized
        self.fc_layers = nn.Sequential()
        for i, size in enumerate(fc_sizes):
            if i == 0:  # first hidden layer
                in_features = env_spec.get('obs_dim')
            else:  # number of out_features from previous layer as in_features for current layer
                in_features = self.fc_layers[i-1].out_features
            # Add FC layer
            self.fc_layers.add_module('fc' + str(i), nn.Linear(in_features, size))
            
        # Action head
        in_features = fc_sizes[-1]
        out_features = env_spec.get('action_dim')
        self.action_head = nn.Linear(in_features, out_features)
        
        # Value head
        if predict_value:
            in_features = fc_sizes[-1]
            out_features = 1
            self.value_head = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # Process input data by user-defined function
        x = self._process_input(x)
        
        # Forward pass till final hidden layer
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        
        # Action head
        action_scores = self.action_head(x)
        action_probs = F.softmax(action_scores, dim=1)
        
        # Value head
        if self.predict_value:
            state_value = self.value_head(x)
        
        # Output dictionary
        output = {}
        output['action_probs'] = action_probs
        if self.predict_value:
            output['state_value'] = state_value
        
        return output
    
    def _process_input(self, x):
        """
        User-defined function to process the input data for the policy network.
        
        Args:
            x (any DType): any DType of input data, it will be processed by the user-defined function _process_input().
            
        Returns:
            out (Tensor): processed input data ready to use for policy network.
        """
        raise NotImplementedError
