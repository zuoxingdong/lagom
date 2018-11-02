import numpy as np

import torch

import pytest

from lagom.history import Transition
from lagom.history import Trajectory
from lagom.history import Segment

from lagom.history.metrics import terminal_state_from_trajectory
from lagom.history.metrics import terminal_state_from_segment
from lagom.history.metrics import final_state_from_trajectory
from lagom.history.metrics import final_state_from_segment
from lagom.history.metrics import bootstrapped_returns_from_trajectory
from lagom.history.metrics import bootstrapped_returns_from_segment


def test_transition():
    transition = Transition(s=1.2, a=2.0, r=-1.0, s_next=1.5, done=True)

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
    assert transition.get_info('V_s') == 0.3
    assert transition.get_info('V_s_next') == 10.0
    assert np.allclose(transition.info['extra'], [1, 2, 3, 4])
    
    
def test_trajectory():
    transition1 = Transition(s=1, a=0.1, r=0.5, s_next=2, done=False)
    transition1.add_info(name='V_s', value=10.0)

    transition2 = Transition(s=2, a=0.2, r=0.5, s_next=3, done=False)
    transition2.add_info(name='V_s', value=20.0)

    transition3 = Transition(s=3, a=0.3, r=1.0, s_next=4, done=True)
    transition3.add_info(name='V_s', value=30.0)
    transition3.add_info(name='V_s_next', value=40.0)  # Note that here non-zero value

    trajectory = Trajectory()

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
    transition4 = Transition(s=0.1, a=0.1, r=1.0, s_next=0.2, done=False)
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
    assert np.allclose(trajectory.all_discounted_returns(0.1), [0.56, 0.6, 1.0])
    assert np.allclose(trajectory.all_info(name='V_s'), [10, 20, 30])
    
    
def test_segment():
    # All test cases with following patterns of values
    # states: 10, 20, ...
    # rewards: 1, 2, ...
    # actions: -1, -2, ...
    # state_value: 100, 200, ...
    # discount: 0.1


    # Test case
    # Part of a single episode
    # [False, False, False, False]
    segment = Segment()

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
    assert np.allclose(segment.all_discounted_returns(0.1), [1.234, 2.34, 3.4, 4])
    
    del segment
    del transition1
    del transition2
    del transition3
    del transition4
    del all_info


    # Test case
    # Part of a single episode with terminal state in final transition
    # [False, False, False, True]
    segment = Segment()

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
    assert np.allclose(segment.all_discounted_returns(0.1), [1.234, 2.34, 3.4, 4])
    
    del segment
    del transition1
    del transition2
    del transition3
    del transition4
    del all_info


    # Test case
    # Two episodes (first episode terminates but second)
    # [False, True, False, False]
    segment = Segment()

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
    assert np.allclose(segment.all_discounted_returns(0.1), [1.2, 2, 3.4, 4])
    
    del segment
    del transition1
    del transition2
    del transition3
    del transition4
    del all_info


    # Test case
    # Three episodes (all terminates)
    # [True, True, False, True]
    segment = Segment()

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
    assert np.allclose(segment.all_discounted_returns(0.1), [1, 2, 3.4, 4])
    
    del segment
    del transition1
    del transition2
    del transition3
    del transition4
    del all_info


def test_terminal_state_from_trajectory():
    t = Trajectory()
    t.add_transition(Transition(1.0, 10, 0.1, 2.0, False))
    t.add_transition(Transition(2.0, 20, 0.2, 3.0, False))
    t.add_transition(Transition(3.0, 30, 0.3, 4.0, True))
    
    assert terminal_state_from_trajectory(t) == 4.0
    
    t = Trajectory()
    t.add_transition(Transition(1.0, 10, 0.1, 2.0, False))
    t.add_transition(Transition(2.0, 20, 0.2, 3.0, False))
    t.add_transition(Transition(3.0, 30, 0.3, 4.0, False))
    
    assert terminal_state_from_trajectory(t) is None
    
    with pytest.raises(AssertionError):
        terminal_state_from_segment(t)

def test_terminal_state_from_segment():
    s = Segment()
    s.add_transition(Transition(1.0, 10, 0.1, 2.0, False))
    s.add_transition(Transition(2.0, 20, 0.2, 3.0, False))
    s.add_transition(Transition(3.0, 30, 0.3, 4.0, True))
    s.add_transition(Transition(5.0, 50, 0.5, 6.0, False))
    s.add_transition(Transition(6.0, 60, 0.6, 7.0, True))
    assert terminal_state_from_segment(s) == [4.0, 7.0]

    s = Segment()
    s.add_transition(Transition(1.0, 10, 0.1, 2.0, False))
    s.add_transition(Transition(2.0, 20, 0.2, 3.0, False))
    s.add_transition(Transition(3.0, 30, 0.3, 4.0, True))
    s.add_transition(Transition(5.0, 50, 0.5, 6.0, False))
    s.add_transition(Transition(6.0, 60, 0.6, 7.0, False))
    assert terminal_state_from_segment(s) == [4.0]
    
    with pytest.raises(AssertionError):
        terminal_state_from_trajectory(s)
        
        
def test_final_state_from_trajectory():
    t = Trajectory()
    t.add_transition(Transition(1.0, 10, 0.1, 2.0, False))
    t.add_transition(Transition(2.0, 20, 0.2, 3.0, False))
    t.add_transition(Transition(3.0, 30, 0.3, 4.0, True))
    
    assert final_state_from_trajectory(t) == 4.0
    
    t = Trajectory()
    t.add_transition(Transition(1.0, 10, 0.1, 2.0, False))
    t.add_transition(Transition(2.0, 20, 0.2, 3.0, False))
    t.add_transition(Transition(3.0, 30, 0.3, 4.0, False))
    
    assert final_state_from_trajectory(t) == 4.0
    
    with pytest.raises(AssertionError):
        final_state_from_segment(t)
        
        
def test_final_state_from_segment():
    s = Segment()
    s.add_transition(Transition(1.0, 10, 0.1, 2.0, False))
    s.add_transition(Transition(2.0, 20, 0.2, 3.0, False))
    s.add_transition(Transition(3.0, 30, 0.3, 4.0, True))
    s.add_transition(Transition(5.0, 50, 0.5, 6.0, False))
    s.add_transition(Transition(6.0, 60, 0.6, 7.0, True))
    assert final_state_from_segment(s) == [4.0, 7.0]

    s = Segment()
    s.add_transition(Transition(1.0, 10, 0.1, 2.0, False))
    s.add_transition(Transition(2.0, 20, 0.2, 3.0, False))
    s.add_transition(Transition(3.0, 30, 0.3, 4.0, True))
    s.add_transition(Transition(5.0, 50, 0.5, 6.0, False))
    s.add_transition(Transition(6.0, 60, 0.6, 7.0, False))
    assert final_state_from_segment(s) == [4.0, 7.0]
    
    with pytest.raises(AssertionError):
        final_state_from_trajectory(s)

        
def test_bootstrapped_returns_from_trajectory():
    t = Trajectory()
    t.add_transition(Transition(1.0, 10, 0.1, 2.0, False))
    t.add_transition(Transition(2.0, 20, 0.2, 3.0, False))
    t.add_transition(Transition(3.0, 30, 0.3, 4.0, True))
    V_last = 100
    
    out =  bootstrapped_returns_from_trajectory(t, V_last, 1.0)
    assert np.allclose(out, [0.6, 0.5, 0.3])
    out =  bootstrapped_returns_from_trajectory(t, V_last, 0.1)
    assert np.allclose(out, [0.123, 0.23, 0.3])
    
    t = Trajectory()
    t.add_transition(Transition(1.0, 10, 0.1, 2.0, False))
    t.add_transition(Transition(2.0, 20, 0.2, 3.0, False))
    t.add_transition(Transition(3.0, 30, 0.3, 4.0, False))
    V_last = 100
    
    out = bootstrapped_returns_from_trajectory(t, V_last, 1.0)
    assert np.allclose(out, [100.6, 100.5, 100.3])
    out = bootstrapped_returns_from_trajectory(t, V_last, 0.1)
    assert np.allclose(out, [0.223, 1.23, 10.3])
    
    with pytest.raises(AssertionError):
        bootstrapped_returns_from_segment(t, V_last, 1.0)
        
        
def test_bootstrapped_returns_from_segment():
    s = Segment()
    s.add_transition(Transition(1.0, 10, 0.1, 2.0, False))
    s.add_transition(Transition(2.0, 20, 0.2, 3.0, False))
    s.add_transition(Transition(3.0, 30, 0.3, 4.0, True))
    s.add_transition(Transition(5.0, 50, 0.5, 6.0, False))
    s.add_transition(Transition(6.0, 60, 0.6, 7.0, True))
    all_V_last = [50, 100]
    
    out = bootstrapped_returns_from_segment(s, all_V_last, 1.0)
    assert np.allclose(out, [0.6, 0.5, 0.3, 1.1, 0.6])
    out = bootstrapped_returns_from_segment(s, all_V_last, 0.1)
    assert np.allclose(out, [0.123, 0.23, 0.3, 0.56, 0.6])

    s = Segment()
    s.add_transition(Transition(1.0, 10, 0.1, 2.0, False))
    s.add_transition(Transition(2.0, 20, 0.2, 3.0, False))
    s.add_transition(Transition(3.0, 30, 0.3, 4.0, True))
    s.add_transition(Transition(5.0, 50, 0.5, 6.0, False))
    s.add_transition(Transition(6.0, 60, 0.6, 7.0, False))
    all_V_last = [50, 100]
    
    out = bootstrapped_returns_from_segment(s, all_V_last, 1.0)
    assert np.allclose(out, [0.6, 0.5, 0.3, 101.1, 100.6])
    out = bootstrapped_returns_from_segment(s, all_V_last, 0.1)
    assert np.allclose(out, [0.123, 0.23, 0.3, 1.56, 10.6])
    
    with pytest.raises(AssertionError):
        bootstrapped_returns_from_trajectory(s, all_V_last, 1.0)
