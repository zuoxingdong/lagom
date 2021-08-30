from typing import List, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
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
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
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

    @jit.export
    def forward_packed(self, input, state):
        # type: (PackedSequence, Tuple[Tensor, Tensor]) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]
        data, batch_sizes, sorted_indices, unsorted_indices = input
        state = (state[0].index_select(0, sorted_indices), state[1].index_select(0, sorted_indices))
        outputs = []
        for batch_size, x in zip(batch_sizes, data.split(batch_sizes.numpy().tolist(), dim=0)):
            assert batch_size == x.shape[0]
            state = (state[0][:batch_size, ...], state[1][:batch_size, ...])
            out, state = self.cell(x, state)
            outputs += [out]
        outputs = PackedSequence(torch.cat(outputs, 0), batch_sizes, sorted_indices, unsorted_indices)
        return outputs, state

    @jit.export
    def forward_tensor(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

    @jit.ignore
    def forward(self, input, state):
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, state)
        else:
            return self.forward_tensor(input, state)


class StackedLSTM(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super().__init__()
        
        self.layers = nn.ModuleList([layer(*first_layer_args)] + [layer(*other_layer_args) 
                                                                  for _ in range(num_layers - 1)])

    @jit.export
    def forward_packed(self, input, states):
        # type: (PackedSequence, List[Tuple[Tensor, Tensor]]) -> Tuple[PackedSequence, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states
        
    @jit.export
    def forward_tensor(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states
    
    @jit.ignore
    def forward(self, input, states):
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, states)
        else:
            return self.forward_tensor(input, states)


def make_lnlstm(input_size, hidden_size, num_layers=1):
    return StackedLSTM(num_layers, 
                       LSTMLayer, 
                       first_layer_args=[LayerNormLSTMCell, input_size, hidden_size], 
                       other_layer_args=[LayerNormLSTMCell, hidden_size, hidden_size])
