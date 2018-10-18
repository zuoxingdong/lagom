import numpy as np

import pytest

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from lagom.networks import ortho_init
from lagom.networks import BaseRNN

from .utils import make_sanity_envs

from lagom.envs import EnvSpec
from lagom.envs import make_gym_env
from lagom.envs import make_envs
from lagom.envs.vec_env import SerialVecEnv

from lagom.agents import BaseAgent
from lagom.agents import RandomAgent

from lagom.history import Transition
from lagom.history import Trajectory
from lagom.history import Segment

from lagom.runner import TrajectoryRunner
from lagom.runner import SegmentRunner


# CartPole-v1
class Agent1(BaseAgent):
    def __init__(self, config, device=None):
        super().__init__(config, device)
        
        self.network = nn.Linear(4, 2)
    
    def choose_action(self, obs, info={}):
        obs = torch.from_numpy(np.array(obs)).float()
        
        action_scores = self.network(obs)
        action_probs = F.softmax(action_scores, dim=-1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)
        
        output = {}
        output['action'] = action
        output['action_logprob'] = action_logprob
        
        return output
        
    def learn(self, D):
        pass
    
    def save(self, filename):
        pass
    
    def load(self, filename):
        pass
    
    
# Pendulum-v0
class Agent2(BaseAgent):
    def __init__(self, config, device=None):
        super().__init__(config, device)
        
        self.network = nn.Linear(3, 1)
    
    def choose_action(self, obs, info={}):
        obs = torch.from_numpy(np.array(obs)).float()
        
        action = self.network(obs)
        action = 2*torch.tanh(action)
        
        output = {}
        output['action'] = action
        output['action_logprob'] = [0.0]*action.shape[0]
        
        return output
        
    def learn(self, D):
        pass
    
    def save(self, filename):
        pass
    
    def load(self, filename):
        pass

    
class TestRunner(object):

    def test_trajectoryrunner(self):
        def check(agent_name, env_name):
            # Create environment
            list_make_env = make_envs(make_env=make_gym_env, 
                                      env_id=env_name, 
                                      num_env=3, 
                                      init_seed=0)
            
            env = SerialVecEnv(list_make_env=list_make_env, rolling=False)
            env_spec = EnvSpec(env)
            
            # Create agent
            if agent_name == 'random':
                agent = RandomAgent(env_spec=env_spec, config=None)
            elif agent_name == 'agent1':
                agent = Agent1(config=None)
            elif agent_name == 'agent2':
                agent = Agent2(config=None)
            else:
                raise ValueError('Wrong agent name')
            
            # Create runner
            runner = TrajectoryRunner(agent=agent, env=env, gamma=1.0)
            with pytest.raises(AssertionError):
                env_fail = SerialVecEnv(list_make_env=list_make_env, rolling=True)
                TrajectoryRunner(agent=agent, env=env_fail, gamma=1.0)

            # Small batch
            D = runner(T=4)

            assert len(D) == 3
            assert all([isinstance(d, Trajectory) for d in D])
            assert all([d.T == 4 for d in D])
            assert all([d.gamma == 1.0 for d in D])

            # Check additional information
            for d in D:
                for t in d.transitions:
                    if agent_name != 'random':
                        assert 'action_logprob' in t.info

            # Check if s in transition is equal to s_next in previous transition
            for d in D:
                for s1, s2 in zip(d.transitions[:-1], d.transitions[1:]):
                    assert np.allclose(s1.s_next, s2.s)
        
            # Long horizon
            D = runner(T=1000)
            for d in D:
                if d.T < 1000:
                    assert d.all_done[-1] == True
            
        # Test for random agent
        # Discrete action space
        check('random', 'CartPole-v1')
        # Continuous action space
        check('random', 'Pendulum-v0')

        # Test for PyTorch agent
        # Discrete action space
        check('agent1', 'CartPole-v1')
        # Continuous action space
        check('agent2', 'Pendulum-v0')

    def test_segmentrunner(self):
        def check(agent_name, env_name):
            # Create environment
            list_make_env = make_envs(make_env=make_gym_env, 
                                      env_id=env_name, 
                                      num_env=2, 
                                      init_seed=0)
            
            env = SerialVecEnv(list_make_env=list_make_env, rolling=True)
            env_spec = EnvSpec(env)
            assert env.num_env == 2

            # Create agent
            if agent_name == 'random':
                agent = RandomAgent(env_spec=env_spec, config=None)
            elif agent_name == 'agent1':
                agent = Agent1(config=None, device=None)
            elif agent_name == 'agent2':
                agent = Agent2(config=None, device=None)
            else:
                raise ValueError('Wrong agent name')

            # Create runner
            runner = SegmentRunner(agent=agent, env=env, gamma=1.0)
            with pytest.raises(AssertionError):
                env_fail = SerialVecEnv(list_make_env=list_make_env, rolling=False)
                SegmentRunner(agent=agent, env=env_fail, gamma=1.0)

            # Small batch
            D = runner(T=3, reset=False)

            assert 'reset_rnn_states' in agent.info and agent.info['reset_rnn_states'] == True
            assert len(D) == 2
            assert all([isinstance(d, Segment) for d in D])
            assert all([d.T == 3 for d in D])
            assert all([d.gamma == 1.0 for d in D])

            # Check additional information
            for d in D:
                for t in d.transitions:
                    if agent_name != 'random':
                        assert 'action_logprob' in t.info

            # Check if s in transition is equal to s_next in previous transition
            for d in D:
                for s1, s2 in zip(d.transitions[:-1], d.transitions[1:]):
                    assert np.allclose(s1.s_next, s2.s)

            # Take one more step, test rolling effect, i.e. first state should be same as last state in previous D
            D2 = runner(T=1, reset=False)
            assert len(D2) == 2
            assert all([d.T == 1 for d in D2])
            for d, d2 in zip(D, D2):
                assert np.allclose(d2.all_s[0][0], d.transitions[-1].s_next)

            # Long horizon
            D = runner(T=200, reset=True)
            # Segment with identical time steps
            assert all([d.T == 200 for d in D])
            # For CartPole, 200 time steps, should be somewhere done=True
            if env_name == 'CartPole-v1':
                assert any([True in d.all_done for d in D])
                assert all([len(d.trajectories) > 1 for d in D])
        
        # Test for random agent
        # Discrete action space
        check('random', 'CartPole-v1')
        # Continuous action space
        check('random', 'Pendulum-v0')

        # Test for PyTorch agent
        # Discrete action space
        check('agent1', 'CartPole-v1')
        # Continuous action space
        check('agent2', 'Pendulum-v0')
        
        
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
    

class RNNAgent(BaseAgent):
    def __init__(self, config, policy, **kwargs):
        super().__init__(config, device=None, **kwargs)
        
        self.policy = policy
        
    def choose_action(self, obs, info={}):
        # Reset RNN states if required
        if self.policy.recurrent and self.info['reset_rnn_states']:
            self.policy.reset_rnn_states()
            self.info['reset_rnn_states'] = False  # Already reset, so turn off
        
        # Convert batched observation
        obs = torch.from_numpy(np.asarray(obs)).float().to(self.policy.device)
        
        out_policy = self.policy(obs, out_keys=['action'], info=info)
        
        
        rnn_states = self.policy.rnn_states
        if isinstance(rnn_states, list):  # LSTM with [h, c]
            # batchize it
            rnn_states = list(zip(*rnn_states))
        
        out_policy['rnn_states'] = rnn_states
        
        return out_policy
        
    def learn(self, D, info={}):
        pass
    
    def save(self, f):
        pass
    
    def load(self, f):
        pass
    
    
def test_trajectory_runner_rnn_agent():
    config = {'network.rnn_size': 16}
    
    env = SerialVecEnv(make_sanity_envs([2, 3]), rolling=False)
    assert env.num_env == 2
    env_spec = EnvSpec(env)
    assert env_spec.num_env == 2
    
    network = LSTM(config=config, env_spec=env_spec)
    assert network.rnn.input_size == 1 and network.rnn.hidden_size == 16
    assert network.last_feature_dim == 16
    assert not hasattr(network, 'action_head')
    assert not hasattr(network, 'value_head')
    
    policy = CategoricalPolicy(config, network, env_spec, 'cpu', learn_V=True)
    assert hasattr(network, 'action_head')
    assert hasattr(network, 'value_head')
    assert policy.recurrent
    rnn_states = policy.rnn_states
    assert isinstance(rnn_states, list) and len(rnn_states) == 2
    h, c = rnn_states
    assert list(h.shape) == [2, 16] and list(c.shape) == [2, 16]
    assert all(np.allclose(i.detach().numpy(), 0.0) for i in [h, c])
    
    agent = RNNAgent(config, policy)
    runner = TrajectoryRunner(agent, env, gamma=1.0)
    
    D = runner(5)
    
    assert len(D) == 2
    assert isinstance(D[0], Trajectory) and isinstance(D[1], Trajectory)
    assert D[0].T == 2 and D[1].T == 3
    
    assert D[0].transitions[0].s == [0.01]
    assert D[0].transitions[0].s_next == [1.01]
    assert D[0].transitions[1].s == [1.01]
    assert D[0].transitions[1].s_next == [2.01]
    assert D[0].transitions[0].r == 0.1
    assert D[0].transitions[1].r == 0.2
    assert not D[0].transitions[0].done
    assert D[0].transitions[1].done
    
    assert D[1].transitions[0].s == [0.01]
    assert D[1].transitions[0].s_next == [1.01]
    assert D[1].transitions[1].s == [1.01]
    assert D[1].transitions[1].s_next == [2.01]
    assert D[1].transitions[2].s == [2.01]
    assert D[1].transitions[2].s_next == [3.01]
    assert D[1].transitions[0].r == 0.1
    assert D[1].transitions[1].r == 0.2
    assert np.allclose(D[1].transitions[2].r, 0.3)
    assert not D[1].transitions[0].done
    assert not D[1].transitions[1].done
    assert D[1].transitions[-1].done
    
    # check RNN states
    h0, c0 = D[0].transitions[0].info['rnn_states']
    assert not np.allclose(h0.detach().numpy(), 0.0) and not np.allclose(c0.detach().numpy(), 0.0)
    assert not h0.equal(c0)
    
    h1, c1 = D[0].transitions[1].info['rnn_states']
    assert not np.allclose(h1.detach().numpy(), 0.0) and not np.allclose(c1.detach().numpy(), 0.0)
    assert not h1.equal(c1)
    assert not h1.equal(h0) and not c1.equal(c0)
    
    h0, c0 = D[1].transitions[0].info['rnn_states']
    assert not np.allclose(h0.detach().numpy(), 0.0) and not np.allclose(c0.detach().numpy(), 0.0)
    assert not h0.equal(c0)
    
    h1, c1 = D[1].transitions[1].info['rnn_states']
    assert not np.allclose(h1.detach().numpy(), 0.0) and not np.allclose(c1.detach().numpy(), 0.0)
    assert not h1.equal(c1)
    assert not h1.equal(h0) and not c1.equal(c0)
    
    h2, c2 = D[1].transitions[2].info['rnn_states']
    assert not np.allclose(h2.detach().numpy(), 0.0) and not np.allclose(c2.detach().numpy(), 0.0)
    assert not h2.equal(c2)
    assert not h2.equal(h1) and not h2.equal(h0)
    assert not c2.equal(c1) and not c2.equal(c0)
    
    
def test_segment_runner_rnn_agent():
    config = {'network.rnn_size': 16}
    
    env = SerialVecEnv(make_sanity_envs([2, 3]), rolling=True)
    assert env.num_env == 2
    env_spec = EnvSpec(env)
    assert env_spec.num_env == 2
    
    network = LSTM(config=config, env_spec=env_spec)
    assert network.rnn.input_size == 1 and network.rnn.hidden_size == 16
    assert network.last_feature_dim == 16
    assert not hasattr(network, 'action_head')
    assert not hasattr(network, 'value_head')
    
    policy = CategoricalPolicy(config, network, env_spec, 'cpu', learn_V=True)
    assert hasattr(network, 'action_head')
    assert hasattr(network, 'value_head')
    assert policy.recurrent
    rnn_states = policy.rnn_states
    assert isinstance(rnn_states, list) and len(rnn_states) == 2
    h, c = rnn_states
    assert list(h.shape) == [2, 16] and list(c.shape) == [2, 16]
    assert all(np.allclose(i.detach().numpy(), 0.0) for i in [h, c])
    
    agent = RNNAgent(config, policy)
    runner = SegmentRunner(agent, env, gamma=1.0)
    
    D = runner(5)
    
    assert len(D) == 2
    assert isinstance(D[0], Segment) and isinstance(D[1], Segment)
    assert D[0].T == 5 and D[1].T == 5
    
    assert D[0].transitions[0].s == [0.01]
    assert D[0].transitions[0].s_next == [1.01]
    assert D[0].transitions[1].s == [1.01]
    assert D[0].transitions[1].s_next == [2.01]
    assert D[0].transitions[2].s == [0.01]
    assert D[0].transitions[2].s_next == [1.01]
    assert D[0].transitions[3].s == [1.01]
    assert D[0].transitions[3].s_next == [2.01]
    assert D[0].transitions[4].s == [0.01]
    assert D[0].transitions[4].s_next == [1.01]
    assert D[0].transitions[0].r == 0.1
    assert D[0].transitions[1].r == 0.2
    assert D[0].transitions[2].r == 0.1
    assert D[0].transitions[3].r == 0.2
    assert D[0].transitions[4].r == 0.1
    assert not D[0].transitions[0].done
    assert D[0].transitions[1].done
    assert not D[0].transitions[2].done
    assert D[0].transitions[3].done
    assert not D[0].transitions[4].done
    
    assert D[1].transitions[0].s == [0.01]
    assert D[1].transitions[0].s_next == [1.01]
    assert D[1].transitions[1].s == [1.01]
    assert D[1].transitions[1].s_next == [2.01]
    assert D[1].transitions[2].s == [2.01]
    assert D[1].transitions[2].s_next == [3.01]
    assert D[1].transitions[3].s == [0.01]
    assert D[1].transitions[3].s_next == [1.01]
    assert D[1].transitions[4].s == [1.01]
    assert D[1].transitions[4].s_next == [2.01]
    assert D[1].transitions[0].r == 0.1
    assert D[1].transitions[1].r == 0.2
    assert np.allclose(D[1].transitions[2].r, 0.3)
    assert D[1].transitions[3].r == 0.1
    assert D[1].transitions[4].r == 0.2
    assert not D[1].transitions[0].done
    assert not D[1].transitions[1].done
    assert D[1].transitions[2].done
    assert not D[1].transitions[3].done
    assert not D[1].transitions[4].done
    
    # check RNN states
    h0, c0 = D[0].transitions[0].info['rnn_states']
    assert not np.allclose(h0.detach().numpy(), 0.0) and not np.allclose(c0.detach().numpy(), 0.0)
    assert not h0.equal(c0)
    
    h1, c1 = D[0].transitions[1].info['rnn_states']
    assert not np.allclose(h1.detach().numpy(), 0.0) and not np.allclose(c1.detach().numpy(), 0.0)
    assert not h1.equal(c1)
    assert not h1.equal(h0) and not c1.equal(c0)
    
    h2, c2 = D[0].transitions[2].info['rnn_states']
    assert not np.allclose(h2.detach().numpy(), 0.0) and not np.allclose(c2.detach().numpy(), 0.0)
    assert not h2.equal(c2)
    assert not h2.equal(h1) and not c2.equal(c1)
    
    h3, c3 = D[0].transitions[3].info['rnn_states']
    assert not np.allclose(h3.detach().numpy(), 0.0) and not np.allclose(c3.detach().numpy(), 0.0)
    assert not h3.equal(c3)
    assert not h3.equal(h2) and not c3.equal(c2)
    
    h4, c4 = D[0].transitions[4].info['rnn_states']
    assert not np.allclose(h4.detach().numpy(), 0.0) and not np.allclose(c4.detach().numpy(), 0.0)
    assert not h4.equal(c4)
    assert not h4.equal(h3) and not c4.equal(c3)
    
    h0, c0 = D[1].transitions[0].info['rnn_states']
    assert not np.allclose(h0.detach().numpy(), 0.0) and not np.allclose(c0.detach().numpy(), 0.0)
    assert not h0.equal(c0)
    
    h1, c1 = D[1].transitions[1].info['rnn_states']
    assert not np.allclose(h1.detach().numpy(), 0.0) and not np.allclose(c1.detach().numpy(), 0.0)
    assert not h1.equal(c1)
    assert not h1.equal(h0) and not c1.equal(c0)
    
    h2, c2 = D[1].transitions[2].info['rnn_states']
    assert not np.allclose(h2.detach().numpy(), 0.0) and not np.allclose(c2.detach().numpy(), 0.0)
    assert not h2.equal(c2)
    assert not h2.equal(h1) and not h2.equal(h0)
    assert not c2.equal(c1) and not c2.equal(c0)
    
    h3, c3 = D[1].transitions[3].info['rnn_states']
    assert not np.allclose(h3.detach().numpy(), 0.0) and not np.allclose(c3.detach().numpy(), 0.0)
    assert not h3.equal(c3)
    assert not h3.equal(h2) and not h3.equal(h1)
    assert not c3.equal(c2) and not c3.equal(c1)
    
    h4, c4 = D[1].transitions[4].info['rnn_states']
    assert not np.allclose(h4.detach().numpy(), 0.0) and not np.allclose(c4.detach().numpy(), 0.0)
    assert not h4.equal(c4)
    assert not h4.equal(h3) and not h4.equal(h2)
    assert not c4.equal(c3) and not c4.equal(c2)
