import numpy as np

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.envs import make_gym_env
from lagom.envs import make_envs
from lagom.envs import EnvSpec
from lagom.envs.vec_env import SerialVecEnv

from lagom.core.networks import BaseNetwork
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


class TestCategoricalPolicy(object):
    def make_env_spec(self):
        list_make_env = make_envs(make_env=make_gym_env, 
                                  env_id='CartPole-v1', 
                                  num_env=1, 
                                  init_seed=0)
        venv = SerialVecEnv(list_make_env=list_make_env)
        env_spec = EnvSpec(venv)
        
        return env_spec
    
    def test_categorical_policy(self):
        env_spec = self.make_env_spec()
        network = Network(env_spec=env_spec)
        
        tmp = CategoricalPolicy(config=None, network=network, env_spec=env_spec)
        assert not hasattr(tmp.network, 'value_head')
        
        policy = CategoricalPolicy(config=None, network=network, env_spec=env_spec, learn_V=True)
        
        assert hasattr(policy, 'config')
        assert hasattr(policy, 'network')
        assert hasattr(policy, 'env_spec')
        
        assert hasattr(policy.network, 'layers')
        assert hasattr(policy.network, 'action_head')
        assert hasattr(policy.network, 'value_head')
        assert len(policy.network.layers) == 1
        assert policy.network.action_head.weight.abs().min().item() <= 0.01  # 0.01 scale for action head
        assert np.allclose(policy.network.action_head.bias.detach().numpy(), 0.0)
        assert policy.network.value_head.weight.abs().max().item() >= 0.1  # roughly +- 0.3 - 0.5
        assert np.allclose(policy.network.value_head.bias.detach().numpy(), 0.0)
        
        obs = torch.from_numpy(np.array(env_spec.env.reset())).float()
        out_policy = policy(obs, out_keys=['action', 'action_prob', 'action_logprob', 
                                           'state_value', 'entropy', 'perplexity'])
        
        assert isinstance(out_policy, dict)
        assert 'action' in out_policy
        assert list(out_policy['action'].shape) == [1]
        assert 'action_prob' in out_policy
        assert list(out_policy['action_prob'].shape) == [1, 2]
        assert 'action_logprob' in out_policy
        assert list(out_policy['action_logprob'].shape) == [1]
        assert 'state_value' in out_policy
        assert list(out_policy['state_value'].shape) == [1]
        assert 'entropy' in out_policy
        assert list(out_policy['entropy'].shape) == [1]
        assert 'perplexity' in out_policy
        assert list(out_policy['perplexity'].shape) == [1]
        
        
class TestGaussianPolicy(object):
    def make_env_spec(self):
        list_make_env = make_envs(make_env=make_gym_env, 
                                  env_id='Pendulum-v0', 
                                  num_env=1, 
                                  init_seed=0)
        venv = SerialVecEnv(list_make_env=list_make_env)
        env_spec = EnvSpec(venv)
        
        return env_spec
    
    def test_gaussian_policy(self):
        env_spec = self.make_env_spec()
        network = Network(env_spec=env_spec)
        
        assert network.num_params == 64
        
        high = np.unique(env_spec.action_space.high).item()
        low = np.unique(env_spec.action_space.low).item()
        
        def _check_policy(policy):
            assert hasattr(policy, 'config')
            assert hasattr(policy, 'network')
            assert hasattr(policy, 'env_spec')
            assert hasattr(policy, 'min_std')
            assert hasattr(policy, 'std_style')
            assert hasattr(policy, 'constant_std')
            assert hasattr(policy, 'std_state_dependent')
            assert hasattr(policy, 'init_std')

            assert hasattr(policy.network, 'layers')
            assert hasattr(policy.network, 'mean_head')
            assert hasattr(policy.network, 'logvar_head')
            assert hasattr(policy.network, 'value_head')
            assert len(policy.network.layers) == 1
            assert policy.network.mean_head.weight.numel() + policy.network.mean_head.bias.numel() == 17
            assert policy.network.mean_head.weight.abs().min().item() <= 0.01  # 0.01 scale for action head
            assert np.allclose(policy.network.mean_head.bias.detach().numpy(), 0.0)
            assert policy.network.value_head.weight.numel() + policy.network.value_head.bias.numel() == 16+1
            assert policy.network.value_head.weight.abs().max().item() >= 0.1  # roughly +- 0.3 - 0.5
            assert np.allclose(policy.network.value_head.bias.detach().numpy(), 0.0)

            obs = torch.from_numpy(np.array(env_spec.env.reset())).float()
            out_policy = policy(obs, out_keys=['action', 'action_logprob', 'state_value', 'entropy', 'perplexity'])

            assert isinstance(out_policy, dict)
            assert 'action' in out_policy
            assert list(out_policy['action'].shape) == [1, 1]
            assert torch.all(out_policy['action'] <= high)
            assert torch.all(out_policy['action'] >= low)
            assert 'action_logprob' in out_policy
            assert list(out_policy['action_logprob'].shape) == [1]
            assert 'state_value' in out_policy
            assert list(out_policy['state_value'].shape) == [1]
            assert 'entropy' in out_policy
            assert list(out_policy['entropy'].shape) == [1]
            assert 'perplexity' in out_policy
            assert list(out_policy['perplexity'].shape) == [1]
        
        # test default without learn_V
        tmp = GaussianPolicy(config=None, 
                             network=network, 
                             env_spec=env_spec)
        assert not hasattr(tmp.network, 'value_head')
        
        # min_std
        network = Network(env_spec=env_spec)
        policy = GaussianPolicy(config=None, 
                                network=network, 
                                env_spec=env_spec, 
                                learn_V=True,
                                min_std=1e-06, 
                                std_style='exp', 
                                constant_std=None, 
                                std_state_dependent=True, 
                                init_std=None)
        _check_policy(policy)
        assert policy.network.num_params - 98 == 17
        assert isinstance(policy.network.logvar_head, nn.Linear)
        assert isinstance(policy.network.value_head, nn.Linear)
        
        # std_style
        network = Network(env_spec=env_spec)
        policy = GaussianPolicy(config=None, 
                                network=network, 
                                env_spec=env_spec, 
                                learn_V=True,
                                min_std=1e-06, 
                                std_style='softplus', 
                                constant_std=None, 
                                std_state_dependent=True, 
                                init_std=None)
        _check_policy(policy)
        assert policy.network.num_params - 98 == 17
        assert isinstance(policy.network.logvar_head, nn.Linear)
        assert isinstance(policy.network.value_head, nn.Linear)
        
        # constant_std
        network = Network(env_spec=env_spec)
        policy = GaussianPolicy(config=None, 
                                network=network, 
                                env_spec=env_spec, 
                                learn_V=True,
                                min_std=1e-06, 
                                std_style='exp', 
                                constant_std=0.1, 
                                std_state_dependent=True, 
                                init_std=None)
        _check_policy(policy)
        assert policy.network.num_params - 98 == 0
        assert torch.is_tensor(policy.network.logvar_head)
        assert policy.network.logvar_head.allclose(torch.tensor(-4.6052))
        
        # std_state_dependent and init_std
        network = Network(env_spec=env_spec)
        policy = GaussianPolicy(config=None, 
                                network=network, 
                                env_spec=env_spec, 
                                learn_V=True,
                                min_std=1e-06, 
                                std_style='exp', 
                                constant_std=None, 
                                std_state_dependent=False, 
                                init_std=0.5)
        _check_policy(policy)
        assert policy.network.num_params - 98 == 1
        assert isinstance(policy.network.logvar_head, nn.Parameter)
        assert policy.network.logvar_head.requires_grad == True
        assert policy.network.logvar_head.allclose(torch.tensor(-1.3863))
