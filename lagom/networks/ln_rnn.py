import torch
import torch.nn as nn
import torch.nn.functional as F

from .make_blocks import make_rnncell


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
