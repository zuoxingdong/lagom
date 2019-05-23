import torch
import torch.nn as nn


class LayerNormLSTMCell(nn.RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, ln_preact=True):
        super(LayerNormLSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)
        
        self.ln_preact = ln_preact
        if self.ln_preact:
            self.ln_ih = nn.LayerNorm(4*self.hidden_size)
            self.ln_hh = nn.LayerNorm(4*self.hidden_size)
        self.ln_cell = nn.LayerNorm(self.hidden_size)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        
        # hidden states and preactivations
        h, c = hx
        ih = input @ self.weight_ih.t() + self.bias_ih
        hh = h @ self.weight_hh.t() + self.bias_hh
        if self.ln_preact:
            ih = self.ln_ih(ih)
            hh = self.ln_hh(hh)
        preact = ih + hh
        
        # Gates
        f, i, o, g = preact.chunk(4, dim=1)
        g = g.tanh()
        f = f.sigmoid()
        i = i.sigmoid()
        o = o.sigmoid()
        
        # cell computations
        c = f*c + i*g
        c = self.ln_cell(c)
        h = o*c.tanh()
        
        return h, c
