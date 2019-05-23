import torch
import torch.nn as nn
import torch.nn.functional as F


def make_rnncell(cell_type, input_dim, hidden_sizes):
    r"""Returns a ModuleList of RNN Cells.
    
    .. note::
    
        All submodules can be automatically tracked because it uses nn.ModuleList. One can
        use this function to generate parameters in :class:`BaseNetwork`. 

    Example::
    
        >>> make_rnncell('LSTMCell', 3, [51, 32, 16])
        ModuleList(
          (0): LSTMCell(3, 51)
          (1): LSTMCell(51, 32)
          (2): LSTMCell(32, 16)
        )
    
    Args:
        cell_type (str): RNNCell type e.g. ['RNNCell', 'LSTMCell', 'GRUCell', 'LayerNormLSTMCell']
        input_dim (int): input dimension in the first recurrent layer. 
        hidden_sizes (list): a list of hidden sizes, each for one recurrent layer. 
    
    Returns
    -------
    rnncell : nn.ModuleList
        a ModuleList of recurrent layers (cells).
        
    """
    assert isinstance(hidden_sizes, list), f'expected as list, got {type(hidden_sizes)}'
    
    if cell_type == 'RNNCell':
        cell_f = nn.RNNCell
    elif cell_type == 'LSTMCell':
        cell_f = nn.LSTMCell
    elif cell_type == 'GRUCell':
        cell_f = nn.GRUCell
    elif cell_type == 'LayerNormLSTMCell':
        cell_f = LayerNormLSTMCell
    else:
        raise ValueError(f'expected RNNCell/LSTMCell/GRUCell/LayerNormLSTMCell, got {cell_type}')
    
    hidden_sizes = [input_dim] + hidden_sizes
    
    rnncell = []
    for input_size, hidden_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
        rnncell.append(cell_f(input_size=input_size, hidden_size=hidden_size))
    
    rnncell = nn.ModuleList(rnncell)
    
    return rnncell


class LayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        if dropout > 0 and num_layers == 1:
            raise ValueError('dropout used for at least two layers, and apply all but last recurrent layer.')
        
        self.ln_cells = make_rnncell('LayerNormLSTMCell', input_size, [hidden_size]*num_layers)
        
    def forward(self, input, hx):
        x = input
        h, c = hx
        for i, ln_cell in enumerate(self.ln_cells):
            output = []
            for t in range(x.size(0)):
                h[i], c[i] = ln_cell(x[t], (h[i].clone(), c[i].clone()))
                output.append(h[i])
            x = torch.stack(output, dim=0)
            if self.dropout > 0 and i < self.num_layers - 1:
                x = F.dropout(x, self.dropout, self.training)
        return x, (h, c)
