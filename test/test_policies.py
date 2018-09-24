import numpy as np

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.envs import make_gym_env
from lagom.envs import make_envs
from lagom.envs import make_vec_env
from lagom.envs import EnvSpec
from lagom.envs.vec_env import SerialVecEnv

from lagom.core.networks import BaseNetwork
from lagom.core.networks import BaseRNN
from lagom.core.networks import make_fc
from lagom.core.networks import ortho_init

from lagom.core.policies import CategoricalPolicy
from lagom.core.policies import GaussianPolicy


class Network(BaseNetwork):
    def make_params(self, config):
        self.layers = make_fc(input_dim=self.env_spec.observation_space.flat_dim, hidden_sizes=[16])
        
        self.last_feature_dim = 16
        
    def init_params(self, config):
        for layer in self.layers:
            ortho_init(layer, nonlinearity='relu')
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
            
        return x
    
    
class LSTM(BaseRNN):
    def make_params(self, config):
        self.rnn = nn.LSTMCell(input_size=self.env_spec.observation_space.flat_dim, 
                               hidden_size=config['network.rnn_size'])

        self.last_feature_dim = config['network.rnn_size']

    def init_params(self, config):
        ortho_init(self.rnn, nonlinearity=None, weight_scale=1.0, constant_bias=0.0)

    def init_hidden_states(self, config, batch_size, **kwargs):
        h = torch.zeros(batch_size, config['network.rnn_size'])
        h = h.to(self.device)
        c = torch.zeros_like(h)

        return [h, c]

    def rnn_forward(self, x, hidden_states, mask=None, **kwargs):
        # mask out hidden states if required
        if mask is not None:
            h, c = hidden_states
            mask = mask.to(self.device)
            
            h = h*mask
            c = c*mask
            hidden_states = [h, c]

        h, c = self.rnn(x, hidden_states)

        out = {'output': h, 'hidden_states': [h, c]}

        return out


class TestCategoricalPolicy(object):
    def make_env_spec(self):
        list_make_env = make_envs(make_env=make_gym_env, 
                                  env_id='CartPole-v1', 
                                  num_env=3, 
                                  init_seed=0)
        venv = SerialVecEnv(list_make_env=list_make_env, rolling=True)
        env_spec = EnvSpec(venv)
        
        return env_spec
    
    @pytest.mark.parametrize('network_type', ['FC', 'LSTM'])
    def test_categorical_policy(self, network_type):
        env_spec = self.make_env_spec()
        device = torch.device('cpu')
        if network_type == 'FC':
            config = {}
            network = Network(config=config, env_spec=env_spec, device=device)
        elif network_type == 'LSTM':
            config = {'network.rnn_size': 16}
            network = LSTM(config=config, env_spec=env_spec, device=device)
        
        
        tmp = CategoricalPolicy(config=config, network=network, env_spec=env_spec, device=device)
        assert not hasattr(tmp.network, 'value_head')
        
        policy = CategoricalPolicy(config=config, network=network, env_spec=env_spec, device=device, learn_V=True)
        
        assert hasattr(policy, 'config')
        assert hasattr(policy, 'network')
        assert hasattr(policy, 'env_spec')
        assert hasattr(policy, 'observation_space')
        assert hasattr(policy, 'action_space')
        assert hasattr(policy, 'device')
        assert hasattr(policy, 'recurrent')
        if network_type == 'FC':
            assert not policy.recurrent
        elif network_type == 'LSTM':
            assert policy.recurrent
            rnn_states = policy.rnn_states
            assert isinstance(rnn_states, list) and len(rnn_states) == 2
            h0, c0 = rnn_states
            assert list(h0.shape) == [3, 16] and list(c0.shape) == list(h0.shape)
            assert np.allclose(h0.detach().numpy(), 0.0)
            assert np.allclose(c0.detach().numpy(), 0.0)
            
        if network_type == 'FC':
            assert hasattr(policy.network, 'layers')
            assert len(policy.network.layers) == 1
        elif network_type == 'LSTM':
            assert hasattr(policy.network, 'rnn')
        assert hasattr(policy.network, 'action_head')
        assert hasattr(policy.network, 'value_head')
        assert hasattr(policy.network, 'device')
        assert policy.network.action_head.weight.abs().min().item() <= 0.01  # 0.01 scale for action head
        assert np.allclose(policy.network.action_head.bias.detach().numpy(), 0.0)
        assert policy.network.value_head.weight.abs().max().item() >= 0.1  # roughly +- 0.3 - 0.5
        assert np.allclose(policy.network.value_head.bias.detach().numpy(), 0.0)
        
        obs = torch.from_numpy(np.array(env_spec.env.reset())).float()
        out_policy = policy(obs, out_keys=['action', 'action_prob', 'action_logprob', 
                                           'state_value', 'entropy', 'perplexity'], info={})
        
        if network_type == 'LSTM':
            new_rnn_states = policy.rnn_states
            assert isinstance(new_rnn_states, list) and len(new_rnn_states) == 2
            h_new, c_new = new_rnn_states
            assert list(h_new.shape) == [3, 16] and list(c_new.shape) == list(h_new.shape)
            assert not np.allclose(h_new.detach().numpy(), 0.0)
            assert not np.allclose(c_new.detach().numpy(), 0.0)
            
            mask = torch.ones(3, 16)*1000
            mask[1] = mask[1].fill_(0.0)
            out_policy=  policy(obs, 
                                out_keys=['action', 'action_prob', 'action_logprob', 
                                          'state_value', 'entropy', 'perplexity'], 
                                info={'mask': mask})
            c = policy.rnn_states[1]
            assert c[0].max().item() >= 1.0 and c[2].max().item() >= 1.0
            assert c[1].max().item() <= 0.1
        
        assert isinstance(out_policy, dict)
        assert 'action' in out_policy
        assert list(out_policy['action'].shape) == [3]
        assert 'action_prob' in out_policy
        assert list(out_policy['action_prob'].shape) == [3, 2]
        assert 'action_logprob' in out_policy
        assert list(out_policy['action_logprob'].shape) == [3]
        assert 'state_value' in out_policy
        assert list(out_policy['state_value'].shape) == [3]
        assert 'entropy' in out_policy
        assert list(out_policy['entropy'].shape) == [3]
        assert 'perplexity' in out_policy
        assert list(out_policy['perplexity'].shape) == [3]
        
        
class TestGaussianPolicy(object):
    def make_env_spec(self):
        list_make_env = make_envs(make_env=make_gym_env, 
                                  env_id='Pendulum-v0', 
                                  num_env=3, 
                                  init_seed=0)
        venv = SerialVecEnv(list_make_env=list_make_env, rolling=True)
        env_spec = EnvSpec(venv)
        
        return env_spec
    
    @pytest.mark.parametrize('network_type', ['FC', 'LSTM'])
    def test_gaussian_policy(self, network_type):
        env_spec = self.make_env_spec()
        device = torch.device('cpu')
        def _create_net(env_spec, device):
            if network_type == 'FC':
                config = {}
                network = Network(config=config, env_spec=env_spec, device=device)
                assert network.num_params == 64
            elif network_type == 'LSTM':
                config = {'network.rnn_size': 16}
                network = LSTM(config=config, env_spec=env_spec, device=device)
                
            return network
        
        if network_type == 'FC':
            config = {}
        elif network_type == 'LSTM':
            config = {'network.rnn_size': 16}
        
        network = _create_net(env_spec, device)
        
        high = np.unique(env_spec.action_space.high).item()
        low = np.unique(env_spec.action_space.low).item()
        
        def _check_policy(policy):
            assert hasattr(policy, 'config')
            assert hasattr(policy, 'network')
            assert hasattr(policy, 'env_spec')
            assert hasattr(policy, 'observation_space')
            assert hasattr(policy, 'action_space')
            assert hasattr(policy, 'device')
            assert hasattr(policy, 'recurrent')
            if network_type == 'FC':
                assert not policy.recurrent
            elif network_type == 'LSTM':
                assert policy.recurrent
                rnn_states = policy.rnn_states
                assert isinstance(rnn_states, list) and len(rnn_states) == 2
                h0, c0 = rnn_states
                assert list(h0.shape) == [3, 16] and list(c0.shape) == list(h0.shape)
                assert np.allclose(h0.detach().numpy(), 0.0)
                assert np.allclose(c0.detach().numpy(), 0.0)
            assert hasattr(policy, 'min_std')
            assert hasattr(policy, 'std_style')
            assert hasattr(policy, 'constant_std')
            assert hasattr(policy, 'std_state_dependent')
            assert hasattr(policy, 'init_std')
            
            if network_type == 'FC':
                assert hasattr(policy.network, 'layers')
                assert len(policy.network.layers) == 1
            elif network_type == 'LSTM':
                assert hasattr(policy.network, 'rnn')
            assert hasattr(policy.network, 'mean_head')
            assert hasattr(policy.network, 'logvar_head')
            assert hasattr(policy.network, 'value_head')
            assert hasattr(policy.network, 'device')
            
            assert policy.network.mean_head.weight.numel() + policy.network.mean_head.bias.numel() == 17
            assert policy.network.mean_head.weight.abs().min().item() <= 0.01  # 0.01 scale for action head
            assert np.allclose(policy.network.mean_head.bias.detach().numpy(), 0.0)
            assert policy.network.value_head.weight.numel() + policy.network.value_head.bias.numel() == 16+1
            assert policy.network.value_head.weight.abs().max().item() >= 0.1  # roughly +- 0.3 - 0.5
            assert np.allclose(policy.network.value_head.bias.detach().numpy(), 0.0)

            obs = torch.from_numpy(np.array(env_spec.env.reset())).float()
            out_policy = policy(obs, 
                                out_keys=['action', 'action_logprob', 'state_value', 'entropy', 'perplexity'], 
                                info={})
            
            if network_type == 'LSTM':
                new_rnn_states = policy.rnn_states
                assert isinstance(new_rnn_states, list) and len(new_rnn_states) == 2
                h_new, c_new = new_rnn_states
                assert list(h_new.shape) == [3, 16] and list(c_new.shape) == [3, 16]
                assert not np.allclose(h_new.detach().numpy(), 0.0)
                assert not np.allclose(c_new.detach().numpy(), 0.0)
                
                mask = torch.ones(3, 16)*1000
                mask[1] = mask[1].fill_(0.0)
                out_policy = policy(obs, 
                                    out_keys=['action', 'action_logprob', 'state_value', 'entropy', 'perplexity'], 
                                    info={'mask': mask})
                c = policy.rnn_states[1]
                assert c[0].max().item() >= 1.0 and c[2].max().item() >= 1.0
                assert c[1].max().item() <= 0.5
                
            assert isinstance(out_policy, dict)
            assert 'action' in out_policy
            assert list(out_policy['action'].shape) == [3, 1]
            assert torch.all(out_policy['action'] <= high)
            assert torch.all(out_policy['action'] >= low)
            assert 'action_logprob' in out_policy
            assert list(out_policy['action_logprob'].shape) == [3]
            assert 'state_value' in out_policy
            assert list(out_policy['state_value'].shape) == [3]
            assert 'entropy' in out_policy
            assert list(out_policy['entropy'].shape) == [3]
            assert 'perplexity' in out_policy
            assert list(out_policy['perplexity'].shape) == [3]
        
        # test default without learn_V
        tmp = GaussianPolicy(config=config, network=network, env_spec=env_spec, device=device)
        assert not hasattr(tmp.network, 'value_head')
        
        # min_std
        network = _create_net(env_spec, device)
        policy = GaussianPolicy(config=config, 
                                network=network, 
                                env_spec=env_spec, 
                                device=device,
                                learn_V=True,
                                min_std=1e-06, 
                                std_style='exp', 
                                constant_std=None, 
                                std_state_dependent=True, 
                                init_std=None)
        _check_policy(policy)
        if network_type == 'FC':
            assert policy.network.num_params - 98 == 17
        assert isinstance(policy.network.logvar_head, nn.Linear)
        assert isinstance(policy.network.value_head, nn.Linear)
        
        # std_style
        network = _create_net(env_spec, device)
        policy = GaussianPolicy(config=config, 
                                network=network, 
                                env_spec=env_spec, 
                                device=device,
                                learn_V=True,
                                min_std=1e-06, 
                                std_style='softplus', 
                                constant_std=None, 
                                std_state_dependent=True, 
                                init_std=None)
        _check_policy(policy)
        if network_type == 'FC':
            assert policy.network.num_params - 98 == 17
        assert isinstance(policy.network.logvar_head, nn.Linear)
        assert isinstance(policy.network.value_head, nn.Linear)
        
        # constant_std
        network = _create_net(env_spec, device)
        policy = GaussianPolicy(config=config, 
                                network=network, 
                                env_spec=env_spec, 
                                device=device,
                                learn_V=True,
                                min_std=1e-06, 
                                std_style='exp', 
                                constant_std=0.1, 
                                std_state_dependent=False, 
                                init_std=None)
        _check_policy(policy)
        if network_type == 'FC':
            assert policy.network.num_params - 98 == 0
        assert torch.is_tensor(policy.network.logvar_head)
        assert policy.network.logvar_head.allclose(torch.tensor(-4.6052))
        
        # std_state_dependent and init_std
        network = _create_net(env_spec, device)
        policy = GaussianPolicy(config=config, 
                                network=network, 
                                env_spec=env_spec, 
                                device=device,
                                learn_V=True,
                                min_std=1e-06, 
                                std_style='exp', 
                                constant_std=None, 
                                std_state_dependent=False, 
                                init_std=0.5)
        _check_policy(policy)
        if network_type == 'FC':
            assert policy.network.num_params - 98 == 1
            assert policy.network.logvar_head.allclose(torch.tensor(-1.3863))
        assert isinstance(policy.network.logvar_head, nn.Parameter)
        assert policy.network.logvar_head.requires_grad == True
