from typing import List, Tuple

import torch
import torch.nn as nn
import torch.jit as jit


class LayerNormLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4*hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4*hidden_size, hidden_size))
        # The layernorms provide learnable biases

        self.layernorm_i = nn.LayerNorm(4*hidden_size)
        self.layernorm_h = nn.LayerNorm(4*hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)

    @jit.script_method
    def forward(self, input, state):
        # (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c(forgetgate*cx + ingate*cellgate)
        hy = outgate*torch.tanh(cy)
        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super().__init__()

        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class StackedLSTM(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super().__init__()
        
        self.layers = nn.ModuleList([layer(*first_layer_args)] + [layer(*other_layer_args) 
                                                                  for _ in range(num_layers - 1)])

    @jit.script_method
    def forward(self, input, states):
        # (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[torch.Tensor, torch.Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states


def make_lnlstm(input_size, hidden_size, num_layers=1):
    return StackedLSTM(num_layers, 
                       LSTMLayer, 
                       first_layer_args=[LayerNormLSTMCell, input_size, hidden_size], 
                       other_layer_args=[LayerNormLSTMCell, hidden_size, hidden_size])
