from lagom.history import Trajectory
from lagom.history import Segment


def final_state_from_trajectory(trajectory):
    r"""Return the final state of a trajectory. 
    
    Args:
        trajectory (Trajectory): a trajectory
        
    Returns
    -------
    out : object
    """
    assert isinstance(trajectory, Trajectory)
    return trajectory.transitions[-1].s_next


def final_state_from_segment(segment):
    r"""Return a list of final states of a segment. 
    
    It collects the final state from each trajectory stored in the segment. 
    
    Args:
        segment (Segment): a segment
        
    Returns
    -------
    out : object
    """
    assert isinstance(segment, Segment)
    
    final_states = []
    for trajectory in segment.trajectories:
        final_states.append(final_state_from_trajectory(trajectory))
        
    return final_states
