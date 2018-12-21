from abc import ABC
from abc import abstractmethod

from .base_network import BaseNetwork


class BaseRNN(BaseNetwork, ABC):
    r"""Base class for all recurrent neural networks. 
    
    Example::
    
        class LSTM(BaseRNN):
            def make_params(self, config):
                self.rnn = nn.LSTMCell(input_size=10, hidden_size=20)

                self.last_feature_dim = 20

            def init_params(self, config):
                ortho_init(self.rnn, nonlinearity=None, weight_scale=1.0, constant_bias=0.0)

            def init_hidden_states(self, config, batch_size, **kwargs):
                h = torch.zeros(batch_size, 20)
                h = h.to(self.device)
                c = torch.zeros_like(h)

                return [h, c]

            def rnn_forward(self, x, hidden_states, **kwargs):
                # mask out hidden states if required
                if 'mask' in kwargs:
                    mask = kwargs['mask'].to(self.device)
                    h, c = hidden_states
                    h = h*mask
                    c = c*mask
                    hidden_states = [h, c]

                h, c = self.rnn(x, hidden_states)

                out = {'output': h, 'hidden_states': [h, c]}

                return out

    """
    @abstractmethod
    def init_hidden_states(self, config, batch_size, **kwargs):
        r"""Returns initialized hidden states for the recurrent neural network. 
        
        Args:
            config (dict): a dictionary of configurations.
            batch_size (int): the batch size for creating the hidden states. 
            **kwargs: keyword aguments used to specify the hidden states initialization
            
        Returns
        -------
        hidden_states : object
            initialized hidden states
        """
        pass
    
    @abstractmethod
    def forward(self, x, hidden_states, **kwargs):
        r"""Defines forward pass for recurrent neural networks. 
        
        Args:
            x (object): input data
            hidden_states (object): hidden states for current time step
            **kwargs: keyword aguments used to specify forward pass. 
            
        Returns
        -------
        out : tuple
            a tuple of forward pass, (output, hidden_states)
        """
        pass
