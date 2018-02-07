import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from lagom.core.policies.mlp_policies import BaseMLPPolicy


class CategoricalMLPPolicy(BaseMLPPolicy):
    def __init__(self, env_spec, fc_sizes=[128], predict_value=False):
        self.env_spec = env_spec
        self.fc_sizes = fc_sizes
        self.predict_value = predict_value
        
        super().__init__(self.env_spec, self.fc_sizes, self.predict_value)
        
    def _process_input(self, x):
        # Unpack input data
        obs = x.get('observation', None)
        
        x = obs
        
        # Convert data into FloatTensor and Variable with batch dimension
        x = Variable(torch.FloatTensor(x).unsqueeze(0))
        
        return x