from lagom.history import Trajectory
from lagom.history import Segment


def terminal_state_from_trajectory(trajectory):
    r"""Return the terminal state of a trajectory if available. 
    
    If the trajectory does not have terminal state, then an ``None`` is returned. 
    
    Args:
        trajectory (Trajectory): a trajectory
        
    Returns
    -------
    out : object
    """
    assert isinstance(trajectory, Trajectory)
    
    if trajectory.complete:
        return trajectory.transitions[-1].s_next
    else:
        return None
    

def terminal_state_from_segment(segment):
    r"""Return a list of terminal states of a segment if available. 
    
    It collects terminal state from each trajectory stored in the segment. 
    
    If the segment does not have terminal state, then an empty list if returned. 
    
    Args:
        segment (Segment): a segment
        
    Returns
    -------
    out : object
    """
    assert isinstance(segment, Segment)
    
    terminal_states = []
    
    for trajectory in segment.trajectories:
        if trajectory.complete:
            terminal_states.append(terminal_state_from_trajectory(trajectory))
    
    return terminal_states
