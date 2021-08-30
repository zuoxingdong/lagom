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

from lagom.nn import Module
from lagom.nn import ortho_init
from lagom.nn import make_fc
from lagom.nn import make_cnn
from lagom.nn import make_transposed_cnn
from lagom.nn import make_lnlstm
from lagom.nn import CategoricalHead
from lagom.nn import DiagGaussianHead
from lagom.nn import bound_logvar


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
@pytest.mark.parametrize('std0', [0.21, 1.0, 3.2, 5.4])
@pytest.mark.parametrize('min_var', [1e-8, 1e-4, 1e-2, 1, 2])
@pytest.mark.parametrize('max_var', [3, 5, 10])
def test_diag_gaussian_head(batch_size, feature_dim, action_dim, std0, min_var, max_var):
    with pytest.raises(AssertionError):
        DiagGaussianHead(feature_dim, action_dim, std_mode=-0.5)
    with pytest.raises(AssertionError):
        DiagGaussianHead(feature_dim, action_dim, std_mode='independent', std0=None)
    with pytest.raises(AssertionError):
        DiagGaussianHead(feature_dim, action_dim, std_mode='independent', std0=0.0)
    with pytest.raises(AssertionError):
        DiagGaussianHead(feature_dim, action_dim, std_mode='dependent', std0=0.1)
    with pytest.raises(AssertionError):
        DiagGaussianHead(feature_dim, action_dim, std_mode='dependent', min_var=None, max_var=max_var)
    with pytest.raises(AssertionError):
        DiagGaussianHead(feature_dim, action_dim, std_mode='dependent', min_var=min_var, max_var=None)
    with pytest.raises(AssertionError):
        DiagGaussianHead(feature_dim, action_dim, std_mode='dependent', min_var=2, max_var=1)
        
    def _basic_check(action_head):
        assert action_head.feature_dim == feature_dim
        assert action_head.action_dim == action_dim
        assert action_head.min_var == min_var
        assert action_head.max_var == max_var
        assert isinstance(action_head.mean_head, nn.Linear)
        if action_head.std_mode == 'independent':
            assert isinstance(action_head.logvar_head, nn.Parameter)
            assert action_head.std0 == std0
        elif action_head.std_mode == 'dependent':
            assert isinstance(action_head.logvar_head, nn.Linear)
            assert action_head.std0 is None
            
    def _dist_check(action_dist):
        assert isinstance(action_dist, Independent)
        assert isinstance(action_dist.base_dist, Normal)
        assert action_dist.batch_shape == (batch_size,)
        assert torch.all(action_dist.variance > min_var)
        assert torch.all(action_dist.variance < max_var)
        action = action_dist.sample()
        assert action.shape == (batch_size, action_dim)
        
    # state-independent std
    action_head = DiagGaussianHead(feature_dim, action_dim, std_mode='independent', std0=std0, min_var=min_var, max_var=max_var)
    _basic_check(action_head)
    action_dist = action_head(torch.randn(batch_size, feature_dim))
    _dist_check(action_dist)
    std0 = torch.exp(0.5*bound_logvar(torch.tensor(std0**2).log(), min_var, max_var))
    assert torch.allclose(action_dist.base_dist.stddev, std0)
    
    # state-dependent std
    action_head = DiagGaussianHead(feature_dim, action_dim, std_mode='dependent', std0=None, min_var=min_var, max_var=max_var)
    _basic_check(action_head)
    action_dist = action_head(torch.randn(batch_size, feature_dim))
    _dist_check(action_dist)
    action_dist = action_head(150.0*torch.randn(batch_size, feature_dim))
    _dist_check(action_dist)


@pytest.mark.parametrize('min_var', [1e-8, 1e-4, 1e-2, 1, 2])
@pytest.mark.parametrize('max_var', [3, 5, 10])
def test_bound_logvar(min_var, max_var):
    logvar = torch.linspace(-100, 100, steps=1000)
    logvar = bound_logvar(logvar, min_var, max_var)
    var = torch.exp(logvar)
    assert np.allclose(var.min(), min_var)
    assert np.allclose(var.max(), max_var)
