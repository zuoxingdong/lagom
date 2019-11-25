import numpy as np

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.optim as optim

from torch.distributions import Categorical
from torch.distributions import Normal
from torch.distributions import Independent

from lagom.networks import Module
from lagom.networks import linear_lr_scheduler
from lagom.networks import ortho_init
from lagom.networks import make_fc
from lagom.networks import make_cnn
from lagom.networks import make_transposed_cnn
from lagom.networks import make_lnlstm
from lagom.networks import CategoricalHead
from lagom.networks import DiagGaussianHead


class TestMakeBlocks(object):
    def test_make_fc(self):
        # Single layer
        fc = make_fc(3, [4])
        assert len(fc) == 1
        
        # Multiple layers
        fc = make_fc(3, [4, 5, 6])
        assert len(fc) == 3
        
        # Raise Exception
        with pytest.raises(AssertionError):
            make_fc(3, 4)
            
    def test_make_cnn(self):
        # Single layer
        cnn = make_cnn(input_channel=3, channels=[16], kernels=[4], strides=[2], paddings=[1])
        assert len(cnn) == 1
        
        # Multiple layers
        cnn = make_cnn(input_channel=3, channels=[16, 32, 64], kernels=[4, 3, 3], strides=[2, 1, 1], paddings=[2, 1, 0])
        assert len(cnn) == 3
        
        # Raise Exception
        with pytest.raises(AssertionError):
            # Non-list
            make_cnn(input_channel=3, channels=[16], kernels=4, strides=[2], paddings=[1])
        with pytest.raises(AssertionError):
            # Inconsistent length
            make_cnn(input_channel=3, channels=[16], kernels=[4, 2], strides=[2], paddings=[1])
            
    def test_make_transposed_cnn(self):
        # Single layer
        transposed_cnn = make_transposed_cnn(input_channel=3, 
                                             channels=[16], 
                                             kernels=[4], 
                                             strides=[2], 
                                             paddings=[1], 
                                             output_paddings=[1])
        assert len(transposed_cnn) == 1
        
        # Multiple layers
        transposed_cnn = make_transposed_cnn(input_channel=3, 
                                             channels=[16, 32, 64], 
                                             kernels=[4, 3, 3], 
                                             strides=[2, 1, 1], 
                                             paddings=[2, 1, 0],
                                             output_paddings=[3, 1, 0])
        assert len(transposed_cnn) == 3
        
        # Raise Exception
        with pytest.raises(AssertionError):
            # Non-list
            make_transposed_cnn(input_channel=3, 
                                channels=[16], 
                                kernels=[4], 
                                strides=2, 
                                paddings=[1], 
                                output_paddings=[1])
        with pytest.raises(AssertionError):
            # Inconsistent length
            make_transposed_cnn(input_channel=3, 
                                channels=[16], 
                                kernels=[4], 
                                strides=[2, 1], 
                                paddings=[1], 
                                output_paddings=[1])
    
    @pytest.mark.parametrize('input_size', [1, 10])
    @pytest.mark.parametrize('hidden_size', [1, 8])
    @pytest.mark.parametrize('batch_size', [1, 5])
    @pytest.mark.parametrize('seq_len', [1, 3])
    def test_make_lnlstm(self, input_size, hidden_size, batch_size, seq_len):
        rnn = make_lnlstm(input_size, hidden_size)
        input = torch.randn(seq_len, batch_size, input_size)
        states = [(torch.randn(batch_size, hidden_size), torch.randn(batch_size, hidden_size))]
        output, output_states = rnn(input, states)
        output, output_states = rnn(input, output_states)
        
        assert output.shape == (seq_len, batch_size, hidden_size)
        hx, cx = output_states[0]
        assert hx.shape == (batch_size, hidden_size)
        assert cx.shape == (batch_size, hidden_size)
        
        # use case: PackedSequence
        x1 = torch.randn(3, input_size)
        x2 = torch.randn(1, input_size)
        x3 = torch.randn(4, input_size)
        x = pad_sequence([x1, x2, x3])
        lengths = [x1.shape[0], x2.shape[0], x3.shape[0]]
        x = pack_padded_sequence(x, lengths, enforce_sorted=False)
        assert x.data.shape == (sum(lengths), input_size)
        h = torch.zeros(3, hidden_size)
        c = torch.zeros_like(h)
        out, [(h, c)] = rnn(x, [(h, c)])
        assert isinstance(out, PackedSequence)
        assert out.data.shape == (sum(lengths), hidden_size)
        assert torch.equal(out.batch_sizes, torch.tensor([3, 2, 2, 1]))
        assert torch.equal(out.sorted_indices, torch.tensor([2, 0, 1]))
        assert torch.equal(out.unsorted_indices, torch.tensor([1, 2, 0]))
        o, l = pad_packed_sequence(out)
        assert torch.is_tensor(o)
        assert o.shape == (4, 3, hidden_size)
        assert torch.equal(l, torch.tensor([3, 1, 4]))
        assert torch.allclose(o[3:, 0, ...], torch.tensor(0.0))
        assert torch.allclose(o[1:, 1, ...], torch.tensor(0.0))
        assert torch.allclose(o[4:, 2, ...], torch.tensor(0.0))


class TestInit(object):
    def test_ortho_init(self):
        # Linear
        a = nn.Linear(2, 3)
        ortho_init(a, weight_scale=1000., constant_bias=10.)
        assert a.weight.max().item() > 30.
        assert np.allclose(a.bias.detach().numpy(), 10.)
        ortho_init(a, nonlinearity='relu')
        
        # Conv2d
        a = nn.Conv2d(2, 3, 3)
        ortho_init(a, weight_scale=1000., constant_bias=10.)
        assert a.weight.max().item() > 50.
        assert np.allclose(a.bias.detach().numpy(), 10.)
        ortho_init(a, nonlinearity='relu')
        
        # LSTM
        a = nn.LSTM(2, 3, 2)
        ortho_init(a, weight_scale=1000., constant_bias=10.)
        assert a.weight_hh_l0.max().item() > 50.
        assert a.weight_hh_l1.max().item() > 50.
        assert a.weight_ih_l0.max().item() > 50.
        assert a.weight_ih_l1.max().item() > 50.
        assert np.allclose(a.bias_hh_l0.detach().numpy(), 10.)
        assert np.allclose(a.bias_hh_l1.detach().numpy(), 10.)
        assert np.allclose(a.bias_ih_l0.detach().numpy(), 10.)
        assert np.allclose(a.bias_ih_l1.detach().numpy(), 10.)
        
        # LSTMCell
        a = nn.LSTMCell(3, 2)
        ortho_init(a, weight_scale=1000., constant_bias=10.)
        assert a.weight_hh.max().item() > 50.
        assert a.weight_ih.max().item() > 50.
        assert np.allclose(a.bias_hh.detach().numpy(), 10.)
        assert np.allclose(a.bias_ih.detach().numpy(), 10.)


@pytest.mark.parametrize('method', ['Adam', 'RMSprop', 'Adamax'])
@pytest.mark.parametrize('N', [1, 10, 50, 100])
@pytest.mark.parametrize('min_lr', [3e-4, 6e-5])
@pytest.mark.parametrize('initial_lr', [1e-3, 7e-4])
def test_linear_lr_scheduler(method, N, min_lr, initial_lr):
    net = nn.Linear(30, 16)
    if method == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=initial_lr)
    elif method == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=initial_lr)
    elif method == 'Adamax':
        optimizer = optim.Adamax(net.parameters(), lr=initial_lr)
    lr_scheduler = linear_lr_scheduler(optimizer, N, min_lr)
    assert lr_scheduler.base_lrs[0] == initial_lr
    
    optimizer.step()
    for i in range(200):
        lr_scheduler.step()
        assert lr_scheduler.get_lr()[0] >= min_lr
    assert lr_scheduler.get_lr()[0] == min_lr       
    
    
@pytest.mark.parametrize('feature_dim', [5, 10, 30])
@pytest.mark.parametrize('batch_size', [1, 16, 32])
@pytest.mark.parametrize('num_action', [1, 4, 10])
def test_categorical_head(feature_dim, batch_size, num_action):
    action_head = CategoricalHead(feature_dim, num_action)
    assert isinstance(action_head, Module)
    assert action_head.feature_dim == feature_dim
    assert action_head.num_action == num_action
    dist = action_head(torch.randn(batch_size, feature_dim))
    assert isinstance(dist, Categorical)
    assert dist.batch_shape == (batch_size,)
    assert dist.probs.shape == (batch_size, num_action)
    x = dist.sample()
    assert x.shape == (batch_size,)
    
    
@pytest.mark.parametrize('batch_size', [1, 32])
@pytest.mark.parametrize('feature_dim', [5, 20])
@pytest.mark.parametrize('action_dim', [1, 4])
@pytest.mark.parametrize('std0', [0.21, 0.5, 1.0])
def test_diag_gaussian_head(batch_size, feature_dim, action_dim, std0):
    with pytest.raises(AssertionError):
        DiagGaussianHead(feature_dim, action_dim, -0.5)
    
    def _basic_check(action_head):
        assert action_head.feature_dim == feature_dim
        assert action_head.action_dim == action_dim
        assert action_head.std0 == std0
        assert isinstance(action_head.mean_head, nn.Linear)
        assert isinstance(action_head.logstd_head, nn.Parameter)
        
    def _dist_check(action_dist):
        assert isinstance(action_dist, Independent)
        assert isinstance(action_dist.base_dist, Normal)
        assert action_dist.batch_shape == (batch_size,)
        action = action_dist.sample()
        assert action.shape == (batch_size, action_dim)
    
    action_head = DiagGaussianHead(feature_dim, action_dim, std0)    
    _basic_check(action_head)
    action_dist = action_head(torch.randn(batch_size, feature_dim))
    _dist_check(action_dist)
    assert torch.allclose(action_dist.base_dist.stddev, torch.tensor(std0))
