import torch.nn as nn
import torch.nn.functional as F

from lagom.core.networks import MLP


class GaussianMLPPolicy(MLP):
    # TODO
    pass
    
    
    
    
    
    
    
    def _process_input(self, x):
        """
        User-defined function to process the input data for the policy network.
        
        Args:
            x (any DType): any DType of input data, it will be processed by the user-defined function _process_input().
            
        Returns:
            out (Tensor): processed input data ready to use for policy network.
        """
        # Unpack input data
        obs = x.get('observation', None)
        
        x = obs
        
        # Convert data into FloatTensor and Variable with batch dimension
        x = Variable(torch.FloatTensor(x).unsqueeze(0))
        
        return x