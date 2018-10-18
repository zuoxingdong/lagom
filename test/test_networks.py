import numpy as np

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.networks import BaseNetwork
from lagom.networks import BaseRNN
from lagom.networks import ortho_init
from lagom.networks import make_fc
from lagom.networks import make_cnn
from lagom.networks import make_transposed_cnn
from lagom.networks import make_rnncell


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
    
    @pytest.mark.parametrize('cell_type', ['RNNCell', 'LSTMCell', 'GRUCell'])
    def test_make_rnncell(self, cell_type):
        # Single layer
        rnn = make_rnncell(cell_type=cell_type, input_dim=3, hidden_sizes=[16])
        assert isinstance(rnn, nn.ModuleList)
        assert all(isinstance(i, nn.RNNCellBase) for i in rnn)
        assert len(rnn) == 1
        
        # Multiple layers
        rnn = make_rnncell(cell_type=cell_type, input_dim=3, hidden_sizes=[16, 32])
        assert isinstance(rnn, nn.ModuleList)
        assert all(isinstance(i, nn.RNNCellBase) for i in rnn)
        assert len(rnn) == 2
        
        # Raise exceptions
        with pytest.raises(ValueError):  # non-defined rnn cell type
            make_rnncell('randomrnn', 3, [16])
        with pytest.raises(AssertionError):  # non-list hidden sizes
            make_rnncell(cell_type, 3, 16)


class TestInit(object):
    def test_ortho_init(self):
        # Linear
        a = nn.Linear(2, 3)
        ortho_init(a, weight_scale=1000., constant_bias=10.)
        assert a.weight.max().item() > 100.
        assert np.allclose(a.bias.detach().numpy(), 10.)
        ortho_init(a, nonlinearity='relu')
        
        # Conv2d
        a = nn.Conv2d(2, 3, 3)
        ortho_init(a, weight_scale=1000., constant_bias=10.)
        assert a.weight.max().item() > 100.
        assert np.allclose(a.bias.detach().numpy(), 10.)
        ortho_init(a, nonlinearity='relu')
        
        # LSTM
        a = nn.LSTM(2, 3, 2)
        ortho_init(a, weight_scale=1000., constant_bias=10.)
        assert a.weight_hh_l0.max().item() > 100.
        assert a.weight_hh_l1.max().item() > 100.
        assert a.weight_ih_l0.max().item() > 100.
        assert a.weight_ih_l1.max().item() > 100.
        assert np.allclose(a.bias_hh_l0.detach().numpy(), 10.)
        assert np.allclose(a.bias_hh_l1.detach().numpy(), 10.)
        assert np.allclose(a.bias_ih_l0.detach().numpy(), 10.)
        assert np.allclose(a.bias_ih_l1.detach().numpy(), 10.)
        
        # LSTMCell
        a = nn.LSTMCell(3, 2)
        ortho_init(a, weight_scale=1000., constant_bias=10.)
        assert a.weight_hh.max().item() > 100.
        assert a.weight_ih.max().item() > 100.
        assert np.allclose(a.bias_hh.detach().numpy(), 10.)
        assert np.allclose(a.bias_ih.detach().numpy(), 10.)
            

class Network(BaseNetwork):
    def make_params(self, config):
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 1)
        
    def init_params(self, config):
        gain = nn.init.calculate_gain('relu')
        
        nn.init.orthogonal_(self.fc1.weight, gain=gain)
        nn.init.constant_(self.fc1.bias, 0.0)
        
        nn.init.orthogonal_(self.fc2.weight, gain=gain)
        nn.init.constant_(self.fc2.bias, 0.0)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def reset(self, config, **kwargs):
        pass
    
    
class LSTM(BaseRNN):
    def make_params(self, config):
        self.rnn = nn.LSTMCell(input_size=10, hidden_size=20)
        
    def init_params(self, config):
        ortho_init(self.rnn, nonlinearity=None, weight_scale=1.0, constant_bias=0.0)
        
    def init_hidden_states(self, config, batch_size, **kwargs):
        h = torch.zeros(batch_size, 20)
        c = torch.zeros_like(h)
        
        return [h, c]
        
    def rnn_forward(self, x, hidden_states, mask=None, **kwargs):
        # mask out hidden states if required
        if mask is not None:
            h, c = hidden_states
            h = h*mask
            c = c*mask
            hidden_states = [h, c]
        
        h, c = self.rnn(x, hidden_states)
        
        out = {'output': h, 'hidden_states': [h, c]}
        
        return out
    
    def reset(self, config, **kwargs):
        pass
    
    
class TestBaseNetwork(object):
    def test_make_params(self):
        net = Network()
        
        net.make_params(config=None)
        
        assert isinstance(net, nn.Module)
        
        assert isinstance(net.fc1, nn.Linear)
        assert net.fc1.in_features == 3
        assert net.fc1.out_features == 2
        
        assert isinstance(net.fc2, nn.Linear)
        assert net.fc2.in_features == 2
        assert net.fc2.out_features == 1
        
    def test_init_params(self):
        net = Network()
        
        net.init_params(config=None)
        
        assert np.allclose(net.fc1.bias.detach().numpy(), 0.0)
        assert np.allclose(net.fc2.bias.detach().numpy(), 0.0)
        
    def test_forward(self):
        net = Network()
        
        x = torch.randn(10, 3)
        y = net(x)
        
        assert list(y.shape) == [10, 1]
        
    def test_num_params(self):
        net = Network()
        
        assert net.num_params == 11
        
    def test_to_vec(self):
        net = Network()
        
        vec = net.to_vec()
        
        assert list(vec.shape) == [11]
        
    def test_from_vec(self):
        net = Network()
        
        net.from_vec(torch.ones(11))
        assert np.allclose(net.to_vec().detach().numpy(), 1.0)

        
class TestBaseRNN(object):
    def test_make_params(self):
        net = LSTM()
        
        assert hasattr(net, 'rnn') and isinstance(net.rnn, nn.RNNCellBase)
        assert net.rnn.input_size == 10 and net.rnn.hidden_size == 20
        assert list(net.rnn.weight_hh.shape) == [80, 20]
        assert list(net.rnn.weight_ih.shape) == [80, 10]
        assert list(net.rnn.bias_hh.shape) == [80]
        assert list(net.rnn.bias_ih.shape) == [80]
        
    def test_init_params(self):
        net = LSTM()
        
        assert np.allclose(net.rnn.bias_hh.detach().numpy(), 0.0)
        assert np.allclose(net.rnn.bias_ih.detach().numpy(), 0.0)
        
    def test_init_hidden_states(self):
        net = LSTM()
        
        h, c = net.init_hidden_states(None, 4)
        
        assert list(h.shape) == [4, 20]
        assert list(c.shape) == [4, 20]
        
    def test_forward(self):
        net = LSTM()
        
        x = torch.randn(2, 3, 10)
        h = torch.ones(3, 20)
        c = torch.ones_like(h)
        net(x[1], [h, c])
        
        mask = torch.ones(3, 20)
        mask[1].fill_(0.0)
        h *= mask
        c *= mask
        net(x[0], [h, c])
        
    def test_num_params(self):
        net = LSTM()
        
        assert net.num_params == 2560
        
    def test_to_vec(self):
        net = LSTM()
        
        vec = net.to_vec()
        
        assert list(vec.shape) == [2560]
        
    def test_from_vec(self):
        net = LSTM()
        
        x = torch.ones(2560)
        
        net.from_vec(x)
        
        assert np.allclose(net.to_vec().detach().numpy(), 1.0)




        

