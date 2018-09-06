import numpy as np

import pytest

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from lagom.envs import EnvSpec, GymEnv
from lagom.agents import BaseAgent, RandomAgent

from lagom.runner import Transition
from lagom.runner import Trajectory
from lagom.runner import TrajectoryRunner
from lagom.runner import Segment
from lagom.runner import SegmentRunner


class Agent1(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        
        self.network = nn.Linear(4, 2)
    
    def choose_action(self, obs):
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
        
    def learn(self, x):
        pass
    
    def save(self, filename):
        pass
    
    def load(self, filename):
        pass
    
    
class Agent2(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        
        self.network = nn.Linear(3, 1)
    
    def choose_action(self, obs):
        obs = torch.from_numpy(np.array(obs)).float()
        
        action = self.network(obs)
        action = 2*torch.tanh(action)
        
        output = {}
        output['action'] = action
        
        return output
        
    def learn(self, x):
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

        assert np.allclose(trajectory.all_s, [1, 2, 3, 4])
        assert np.allclose(trajectory.all_a, [0.1, 0.2, 0.3])
        assert np.allclose(trajectory.all_r, [0.5, 0.5, 1.0])
        assert np.allclose(trajectory.all_done, [False, False, True])
        assert np.allclose(trajectory.all_returns, [2.0, 1.5, 1.0])
        assert np.allclose(trajectory.all_discounted_returns, [0.56, 0.6, 1.0])
        assert np.allclose(trajectory.all_bootstrapped_returns, [2.0, 1.5, 1.0])
        assert np.allclose(trajectory.all_bootstrapped_discounted_returns, [0.56, 0.6, 1.0])
        assert np.allclose(trajectory.all_V, [10, 20, 30, 40])
        assert np.allclose(trajectory.all_TD, [-7.5, -16.5, -29])
        assert np.allclose(trajectory.all_info(name='V_s'), [10, 20, 30])
        
        # Make last transition: done=False
        trajectory.transitions[-1].done = False
        assert np.allclose(trajectory.all_done, [False, False, False])
        assert np.allclose(trajectory.all_bootstrapped_returns, [42, 41.5, 41])
        assert np.allclose(trajectory.all_bootstrapped_discounted_returns, [0.6, 1, 5])
        assert np.allclose(trajectory.all_V, [10, 20, 30, 40])
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

        assert np.allclose(segment.all_s, [10, 20, 30, 40, 50])
        assert np.allclose(segment.all_a, [-1, -2, -3, -4])
        assert np.allclose(segment.all_r, [1, 2, 3, 4])
        assert np.allclose(segment.all_done, [False, False, False, False])
        assert np.allclose(segment.all_returns, [10, 9, 7, 4])
        assert np.allclose(segment.all_discounted_returns, [1.234, 2.34, 3.4, 4])
        assert np.allclose(segment.all_bootstrapped_returns, [510, 509, 507, 504])
        assert np.allclose(segment.all_bootstrapped_discounted_returns, [1.284, 2.84, 8.4, 54])
        assert segment.all_V == [torch.tensor(i) for i in [100., 200., 300., 400., 500.]]
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

        assert np.allclose(segment.all_s, [10, 20, 30, 40, 50])
        assert np.allclose(segment.all_a, [-1, -2, -3, -4])
        assert np.allclose(segment.all_r, [1, 2, 3, 4])
        assert np.allclose(segment.all_done, [False, False, False, True])
        assert np.allclose(segment.all_returns, [10, 9, 7, 4])
        assert np.allclose(segment.all_discounted_returns, [1.234, 2.34, 3.4, 4])
        assert np.allclose(segment.all_bootstrapped_returns, [10, 9, 7, 4])
        assert np.allclose(segment.all_bootstrapped_discounted_returns, [1.234, 2.34, 3.4, 4])
        assert segment.all_V == [torch.tensor(i) for i in [100., 200., 300., 400., 500.]]
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

        assert np.allclose(segment.all_s, [10, 20, 30, 35, 40, 50])
        assert np.allclose(segment.all_a, [-1, -2, -3, -4])
        assert np.allclose(segment.all_r, [1, 2, 3, 4])
        assert np.allclose(segment.all_done, [False, True, False, False])
        assert np.allclose(segment.all_returns, [3, 2, 7, 4])
        assert np.allclose(segment.all_discounted_returns, [1.2, 2, 3.4, 4])
        assert np.allclose(segment.all_bootstrapped_returns, [3, 2, 507, 504])
        assert np.allclose(segment.all_bootstrapped_discounted_returns, [1.2, 2, 8.4, 54])
        assert segment.all_V == [torch.tensor(i) for i in [100., 200., 250., 300., 400., 500.]]
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

        assert np.allclose(segment.all_s, [10, 20, 25, 30, 35, 40, 50])
        assert np.allclose(segment.all_a, [-1, -2, -3, -4])
        assert np.allclose(segment.all_r, [1, 2, 3, 4])
        assert np.allclose(segment.all_done, [True, True, False, True])
        assert np.allclose(segment.all_returns, [1, 2, 7, 4])
        assert np.allclose(segment.all_discounted_returns, [1, 2, 3.4, 4])
        assert np.allclose(segment.all_bootstrapped_returns, [1, 2, 7, 4])
        assert np.allclose(segment.all_bootstrapped_discounted_returns, [1, 2, 3.4, 4])
        assert segment.all_V == [torch.tensor(i) for i in [100., 150., 200., 250., 300., 400., 500.]]
        assert np.allclose(segment.all_TD, [-99, -198, -257, -396])

        del segment
        del transition1
        del transition2
        del transition3
        del transition4
        del all_info
        
    def test_trajectoryrunner(self):
        def helper(agent, env):
            env = GymEnv(env)
            env_spec = EnvSpec(env)

            if agent == 'random':
                agent = RandomAgent(env_spec=env_spec, config=None)
            elif agent == 'agent1':
                agent = Agent1(config=None)
            elif agent == 'agent2':
                agent = Agent2(config=None)
            else:
                raise ValueError('Wrong agent name')

            runner = TrajectoryRunner(agent=agent, env=env, gamma=1.0)

            D = runner(N=3, T=4)

            assert len(D) == 3
            assert np.alltrue([isinstance(d, Trajectory) for d in D])
            assert np.alltrue([d.T == 4 for d in D])
            assert np.alltrue([d.gamma == runner.gamma for d in D])
            
        # Test for random agent
        # Discrete action space
        env = gym.make('CartPole-v1')
        helper('random', env)
        # Continuous action space
        env = gym.make('Pendulum-v0')
        helper('random', env)

        # Test for PyTorch agent
        # Discrete action space
        env = gym.make('CartPole-v1')
        helper('agent1', env)
        # Continuous action space
        env = gym.make('Pendulum-v0')
        helper('agent2', env)
