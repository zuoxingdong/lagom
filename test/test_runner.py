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
from lagom.runner import Runner


class Agent1(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        
        self.network = nn.Linear(4, 2)
    
    def choose_action(self, obs):
        obs = torch.from_numpy(obs).float()
        obs = obs.unsqueeze(0)
        
        action_scores = self.network(obs)
        action_probs = F.softmax(action_scores, dim=-1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)
        
        action = action.squeeze(0)
        
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
        obs = torch.from_numpy(obs).float()
        obs = obs.unsqueeze(0)
        
        action = self.network(obs)
        action = 2*torch.tanh(action)
        action = action.squeeze(0)
        
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
        assert transition.done
        
        assert len(transition.info) == 0
        
        transition.add_info(name='V_s', value=0.3)
        transition.add_info(name='V_s_next', value=0.0)
        transition.add_info(name='extra', value=[1, 2, 3, 4])
        
        assert len(transition.info) == 3
        assert transition.V_s == 0.3
        assert transition.V_s_next == 0.0
        assert np.allclose(transition.info['extra'], [1, 2, 3, 4])
        
    def test_trajectory(self):
        transition1 = Transition(s=1, 
                                 a=0.1, 
                                 r=0.5, 
                                 s_next=2, 
                                 done=False)
        transition1.add_info(name='V_s', value=10)

        transition2 = Transition(s=2, 
                                 a=0.2, 
                                 r=0.5, 
                                 s_next=3, 
                                 done=False)
        transition2.add_info(name='V_s', value=20)

        transition3 = Transition(s=3, 
                                 a=0.3, 
                                 r=1.0, 
                                 s_next=4, 
                                 done=True)
        transition3.add_info(name='V_s', value=30)
        transition3.add_info(name='V_s_next', value=0.0)

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

        assert np.allclose(trajectory.all_s, [1, 2, 3, 4])
        assert np.allclose(trajectory.all_a, [0.1, 0.2, 0.3])
        assert np.allclose(trajectory.all_r, [0.5, 0.5, 1.0])
        assert np.allclose(trajectory.all_done, [False, False, True])
        assert np.allclose(trajectory.all_returns, [2.0, 1.5, 1.0])
        assert np.allclose(trajectory.all_discounted_returns, [0.56, 0.6, 1.0])
        assert np.allclose(trajectory.all_V, [10, 20, 30, 0])
        assert np.allclose(trajectory.all_TD, [-7.5, -16.5, -29])
        assert np.allclose(trajectory.all_info(name='V_s'), [10, 20, 30])
        
    def test_runner(self):
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

            runner = Runner(agent=agent, env=env, gamma=1.0)

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
