import numpy as np

import pytest

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from lagom.core.networks import ortho_init
from lagom.core.networks import BaseRNN

from lagom.core.policies import CategoricalPolicy
from lagom.core.policies import GaussianPolicy

from .utils import make_sanity_envs

from lagom.envs import EnvSpec
from lagom.envs import make_gym_env
from lagom.envs import make_envs
from lagom.envs.vec_env import SerialVecEnv

from lagom.agents import BaseAgent
from lagom.agents import RandomAgent

from lagom.runner import Transition
from lagom.runner import Trajectory
from lagom.runner import TrajectoryRunner
from lagom.runner import Segment
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
    def test_transition(self):
        transition = Transition(s=1.2, 
                                a=2.0, 
                                r=-1.0, 
                                s_next=1.5, 
                                done=True)
        
        assert transition.s == 1.2
        assert transition.a == 2.0
        assert transition.r == -1.0
        assert transition.s_next == 1.5
        assert transition.done == True
        
        assert len(transition.info) == 0
        
        transition.add_info(name='V_s', value=0.3)
        transition.add_info(name='V_s_next', value=10.0)
        transition.add_info(name='extra', value=[1, 2, 3, 4])
        
        assert len(transition.info) == 3
        assert transition.V_s == 0.3
        assert transition.V_s_next == 10.0
        assert np.allclose(transition.info['extra'], [1, 2, 3, 4])
        
    def test_trajectory(self):
        transition1 = Transition(s=1, 
                                 a=0.1, 
                                 r=0.5, 
                                 s_next=2, 
                                 done=False)
        transition1.add_info(name='V_s', value=10.0)

        transition2 = Transition(s=2, 
                                 a=0.2, 
                                 r=0.5, 
                                 s_next=3, 
                                 done=False)
        transition2.add_info(name='V_s', value=20.0)

        transition3 = Transition(s=3, 
                                 a=0.3, 
                                 r=1.0, 
                                 s_next=4, 
                                 done=True)
        transition3.add_info(name='V_s', value=30.0)
        transition3.add_info(name='V_s_next', value=40.0)  # Note that here non-zero value

        trajectory = Trajectory(gamma=0.1)

        assert trajectory.gamma == 0.1
        assert len(trajectory.info) == 0
        assert trajectory.T == 0

        trajectory.add_info(name='extra', value=[1, 2, 3])
        assert len(trajectory.info) == 1
        assert np.allclose(trajectory.info['extra'], [1, 2, 3])

        trajectory.add_transition(transition=transition1)
        trajectory.add_transition(transition=transition2)
        trajectory.add_transition(transition=transition3)

        assert trajectory.T == 3
        
        # Test error to add one more transition, not allowed because last transition already done=True
        transition4 = Transition(s=0.1, 
                                 a=0.1, 
                                 r=1.0, 
                                 s_next=0.2, 
                                 done=False)
        with pytest.raises(AssertionError):
            trajectory.add_transition(transition=transition4)
        
        all_s = trajectory.all_s
        assert isinstance(all_s, tuple) and len(all_s) == 2
        assert np.allclose(all_s[0], [1, 2, 3])
        assert all_s[1] == 4
        assert np.allclose(trajectory.all_a, [0.1, 0.2, 0.3])
        assert np.allclose(trajectory.all_r, [0.5, 0.5, 1.0])
        assert np.allclose(trajectory.all_done, [False, False, True])
        assert np.allclose(trajectory.all_returns, [2.0, 1.5, 1.0])
        assert np.allclose(trajectory.all_discounted_returns, [0.56, 0.6, 1.0])
        assert np.allclose(trajectory.all_bootstrapped_returns, [2.0, 1.5, 1.0])
        assert np.allclose(trajectory.all_bootstrapped_discounted_returns, [0.56, 0.6, 1.0])
        all_V = trajectory.all_V
        assert isinstance(all_V, tuple) and len(all_V) == 2
        assert np.allclose(all_V[0], [10, 20, 30])
        assert isinstance(all_V[1], list) and len(all_V[1]) == 2
        assert all_V[1][0] == 40 and all_V[1][1]
        assert np.allclose(trajectory.all_TD, [-7.5, -16.5, -29])
        assert np.allclose(trajectory.all_info(name='V_s'), [10, 20, 30])
        
        # Make last transition: done=False
        trajectory.transitions[-1].done = False
        assert np.allclose(trajectory.all_done, [False, False, False])
        assert np.allclose(trajectory.all_bootstrapped_returns, [42, 41.5, 41])
        assert np.allclose(trajectory.all_bootstrapped_discounted_returns, [0.6, 1, 5])
        all_V = trajectory.all_V
        assert isinstance(all_V, tuple) and len(all_V) == 2
        assert np.allclose(all_V[0], [10, 20, 30])
        assert isinstance(all_V[1], list) and len(all_V[1]) == 2
        assert all_V[1][0] == 40 and not all_V[1][1]
        assert np.allclose(trajectory.all_TD, [-7.5, -16.5, -25])
        
    def test_segment(self):
        # All test cases with following patterns of values
        # states: 10, 20, ...
        # rewards: 1, 2, ...
        # actions: -1, -2, ...
        # state_value: 100, 200, ...
        # discount: 0.1


        # Test case
        # Part of a single episode
        # [False, False, False, False]
        segment = Segment(gamma=0.1)

        transition1 = Transition(s=10, a=-1, r=1, s_next=20, done=False)
        transition1.add_info('V_s', torch.tensor(100.))
        segment.add_transition(transition1)

        transition2 = Transition(s=20, a=-2, r=2, s_next=30, done=False)
        transition2.add_info('V_s', torch.tensor(200.))
        segment.add_transition(transition2)

        transition3 = Transition(s=30, a=-3, r=3, s_next=40, done=False)
        transition3.add_info('V_s', torch.tensor(300.))
        segment.add_transition(transition3)

        transition4 = Transition(s=40, a=-4, r=4, s_next=50, done=False)
        transition4.add_info('V_s', torch.tensor(400.))
        transition4.add_info('V_s_next', torch.tensor(500.))
        assert len(transition4.info) == 2
        segment.add_transition(transition4)

        segment.add_info('extra', 'ok')
        assert len(segment.info) == 1

        # all_info
        all_info = segment.all_info('V_s')
        assert all([torch.is_tensor(info) for info in all_info])
        assert all_info[0].item() == 100.0
        assert all_info[-1].item() == 400.0

        assert segment.gamma == 0.1
        assert all([trajectory.gamma == 0.1 for trajectory in segment.trajectories])
        assert segment.T == 4
        assert len(segment.trajectories) == 1
        assert segment.trajectories[0].T == 4

        all_s = segment.all_s
        assert isinstance(all_s, tuple) and len(all_s) == 2
        assert np.allclose(all_s[0], [10, 20, 30, 40])
        assert isinstance(all_s[1], tuple) and len(all_s[1]) == 1
        assert all_s[1][0] == 50
        assert np.allclose(segment.all_a, [-1, -2, -3, -4])
        assert np.allclose(segment.all_r, [1, 2, 3, 4])
        assert np.allclose(segment.all_done, [False, False, False, False])
        assert np.allclose(segment.all_returns, [10, 9, 7, 4])
        assert np.allclose(segment.all_discounted_returns, [1.234, 2.34, 3.4, 4])
        assert np.allclose(segment.all_bootstrapped_returns, [510, 509, 507, 504])
        assert np.allclose(segment.all_bootstrapped_discounted_returns, [1.284, 2.84, 8.4, 54])
        all_V = segment.all_V
        assert isinstance(all_V, tuple) and len(all_V) == 2
        assert all_V[0] == [torch.tensor(i) for i in [100., 200., 300., 400.]]
        assert isinstance(all_V[1], tuple) and len(all_V[1]) == 1
        assert torch.is_tensor(all_V[1][0][0]) and all_V[1][0][0].item() == 500. and not all_V[1][0][1]
        assert np.allclose(segment.all_TD, [-79, -168, -257, -346])

        del segment
        del transition1
        del transition2
        del transition3
        del transition4
        del all_info


        # Test case
        # Part of a single episode with terminal state in final transition
        # [False, False, False, True]
        segment = Segment(gamma=0.1)

        transition1 = Transition(s=10, a=-1, r=1, s_next=20, done=False)
        transition1.add_info('V_s', torch.tensor(100.))
        segment.add_transition(transition1)

        transition2 = Transition(s=20, a=-2, r=2, s_next=30, done=False)
        transition2.add_info('V_s', torch.tensor(200.))
        segment.add_transition(transition2)

        transition3 = Transition(s=30, a=-3, r=3, s_next=40, done=False)
        transition3.add_info('V_s', torch.tensor(300.))
        segment.add_transition(transition3)

        transition4 = Transition(s=40, a=-4, r=4, s_next=50, done=True)
        transition4.add_info('V_s', torch.tensor(400.))
        transition4.add_info('V_s_next', torch.tensor(500.))
        assert len(transition4.info) == 2
        segment.add_transition(transition4)

        segment.add_info('extra', 'ok')
        assert len(segment.info) == 1

        # all_info
        all_info = segment.all_info('V_s')
        assert all([torch.is_tensor(info) for info in all_info])
        assert all_info[0].item() == 100.0
        assert all_info[-1].item() == 400.0

        assert segment.gamma == 0.1
        assert all([trajectory.gamma == 0.1 for trajectory in segment.trajectories])
        assert segment.T == 4
        assert len(segment.trajectories) == 1
        assert segment.trajectories[0].T == 4
        assert len(segment.transitions) == 4

        all_s = segment.all_s
        assert isinstance(all_s, tuple) and len(all_s) == 2
        assert np.allclose(all_s[0], [10, 20, 30, 40])
        assert isinstance(all_s[1], tuple) and len(all_s[1]) == 1
        assert all_s[1][0] == 50
        assert np.allclose(segment.all_a, [-1, -2, -3, -4])
        assert np.allclose(segment.all_r, [1, 2, 3, 4])
        assert np.allclose(segment.all_done, [False, False, False, True])
        assert np.allclose(segment.all_returns, [10, 9, 7, 4])
        assert np.allclose(segment.all_discounted_returns, [1.234, 2.34, 3.4, 4])
        assert np.allclose(segment.all_bootstrapped_returns, [10, 9, 7, 4])
        assert np.allclose(segment.all_bootstrapped_discounted_returns, [1.234, 2.34, 3.4, 4])
        all_V = segment.all_V
        assert isinstance(all_V, tuple) and len(all_V) == 2
        assert all_V[0] == [torch.tensor(i) for i in [100., 200., 300., 400.]]
        assert isinstance(all_V[1], tuple) and len(all_V[1]) == 1
        assert torch.is_tensor(all_V[1][0][0]) and all_V[1][0][0].item() == 500. and all_V[1][0][1]
        assert np.allclose(segment.all_TD, [-79, -168, -257, -396])

        del segment
        del transition1
        del transition2
        del transition3
        del transition4
        del all_info


        # Test case
        # Two episodes (first episode terminates but second)
        # [False, True, False, False]
        segment = Segment(gamma=0.1)

        transition1 = Transition(s=10, a=-1, r=1, s_next=20, done=False)
        transition1.add_info('V_s', torch.tensor(100.))
        segment.add_transition(transition1)

        transition2 = Transition(s=20, a=-2, r=2, s_next=30, done=True)
        transition2.add_info('V_s', torch.tensor(200.))
        transition2.add_info('V_s_next', torch.tensor(250.))
        assert len(transition2.info) == 2
        segment.add_transition(transition2)

        transition3 = Transition(s=35, a=-3, r=3, s_next=40, done=False)
        transition3.add_info('V_s', torch.tensor(300.))
        segment.add_transition(transition3)

        transition4 = Transition(s=40, a=-4, r=4, s_next=50, done=False)
        transition4.add_info('V_s', torch.tensor(400.))
        transition4.add_info('V_s_next', torch.tensor(500.))
        assert len(transition4.info) == 2
        segment.add_transition(transition4)

        segment.add_info('extra', 'ok')
        assert len(segment.info) == 1

        # all_info
        all_info = segment.all_info('V_s')
        assert all([torch.is_tensor(info) for info in all_info])
        assert all_info[0].item() == 100.0
        assert all_info[-1].item() == 400.0

        assert segment.gamma == 0.1
        assert all([trajectory.gamma == 0.1 for trajectory in segment.trajectories])
        assert segment.T == 4
        assert len(segment.trajectories) == 2
        assert segment.trajectories[0].T == 2
        assert segment.trajectories[1].T == 2
        assert len(segment.transitions) == 4

        all_s = segment.all_s
        assert isinstance(all_s, tuple) and len(all_s) == 2
        assert np.allclose(all_s[0], [10, 20, 35, 40])
        assert isinstance(all_s[1], tuple) and len(all_s[1]) == 2
        assert all_s[1] == (30, 50)
        assert np.allclose(segment.all_a, [-1, -2, -3, -4])
        assert np.allclose(segment.all_r, [1, 2, 3, 4])
        assert np.allclose(segment.all_done, [False, True, False, False])
        assert np.allclose(segment.all_returns, [3, 2, 7, 4])
        assert np.allclose(segment.all_discounted_returns, [1.2, 2, 3.4, 4])
        assert np.allclose(segment.all_bootstrapped_returns, [3, 2, 507, 504])
        assert np.allclose(segment.all_bootstrapped_discounted_returns, [1.2, 2, 8.4, 54])
        all_V = segment.all_V
        assert isinstance(all_V, tuple) and len(all_V) == 2
        assert all_V[0] == [torch.tensor(i) for i in [100., 200., 300., 400.]]
        assert isinstance(all_V[1], tuple) and len(all_V[1]) == 2
        assert torch.is_tensor(all_V[1][0][0]) and all_V[1][0][0].item() == 250. and all_V[1][0][1]
        assert torch.is_tensor(all_V[1][1][0]) and all_V[1][1][0].item() == 500. and not all_V[1][1][1]
        assert np.allclose(segment.all_TD, [-79, -198, -257, -346])

        del segment
        del transition1
        del transition2
        del transition3
        del transition4
        del all_info


        # Test case
        # Three episodes (all terminates)
        # [True, True, False, True]
        segment = Segment(gamma=0.1)

        transition1 = Transition(s=10, a=-1, r=1, s_next=20, done=True)
        transition1.add_info('V_s', torch.tensor(100.))
        transition1.add_info('V_s_next', torch.tensor(150.))
        assert len(transition1.info) == 2
        segment.add_transition(transition1)

        transition2 = Transition(s=25, a=-2, r=2, s_next=30, done=True)
        transition2.add_info('V_s', torch.tensor(200.))
        transition2.add_info('V_s_next', torch.tensor(250.))
        assert len(transition2.info) == 2
        segment.add_transition(transition2)

        transition3 = Transition(s=35, a=-3, r=3, s_next=40, done=False)
        transition3.add_info('V_s', torch.tensor(300.))
        segment.add_transition(transition3)

        transition4 = Transition(s=40, a=-4, r=4, s_next=50, done=True)
        transition4.add_info('V_s', torch.tensor(400.))
        transition4.add_info('V_s_next', torch.tensor(500.))
        assert len(transition4.info) == 2
        segment.add_transition(transition4)

        segment.add_info('extra', 'ok')
        assert len(segment.info) == 1

        # all_info
        all_info = segment.all_info('V_s')
        assert all([torch.is_tensor(info) for info in all_info])
        assert all_info[0].item() == 100.0
        assert all_info[-1].item() == 400.0

        assert segment.gamma == 0.1
        assert all([trajectory.gamma == 0.1 for trajectory in segment.trajectories])
        assert segment.T == 4
        assert len(segment.trajectories) == 3
        assert segment.trajectories[0].T == 1
        assert segment.trajectories[1].T == 1
        assert segment.trajectories[2].T == 2
        assert len(segment.transitions) == 4

        all_s = segment.all_s
        assert isinstance(all_s, tuple) and len(all_s) == 2
        assert np.allclose(all_s[0], [10, 25, 35, 40])
        assert isinstance(all_s[1], tuple) and len(all_s[1]) == 3
        assert all_s[1] == (20, 30, 50)
        assert np.allclose(segment.all_a, [-1, -2, -3, -4])
        assert np.allclose(segment.all_r, [1, 2, 3, 4])
        assert np.allclose(segment.all_done, [True, True, False, True])
        assert np.allclose(segment.all_returns, [1, 2, 7, 4])
        assert np.allclose(segment.all_discounted_returns, [1, 2, 3.4, 4])
        assert np.allclose(segment.all_bootstrapped_returns, [1, 2, 7, 4])
        assert np.allclose(segment.all_bootstrapped_discounted_returns, [1, 2, 3.4, 4])
        all_V = segment.all_V
        assert isinstance(all_V, tuple) and len(all_V) == 2
        assert all_V[0] == [torch.tensor(i) for i in [100., 200., 300., 400.]]
        assert isinstance(all_V[1], tuple) and len(all_V[1]) == 3
        assert torch.is_tensor(all_V[1][0][0]) and all_V[1][0][0].item() == 150. and all_V[1][0][1]
        assert torch.is_tensor(all_V[1][1][0]) and all_V[1][1][0].item() == 250. and all_V[1][1][1]
        assert torch.is_tensor(all_V[1][2][0]) and all_V[1][2][0].item() == 500. and all_V[1][2][1]
        assert np.allclose(segment.all_TD, [-99, -198, -257, -396])

        del segment
        del transition1
        del transition2
        del transition3
        del transition4
        del all_info
        
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
