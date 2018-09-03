import numpy as np

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.core.networks import BaseNetwork
from lagom.core.networks import ortho_init
from lagom.core.networks import make_fc
from lagom.core.networks import make_cnn
from lagom.core.networks import make_transposed_cnn


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
