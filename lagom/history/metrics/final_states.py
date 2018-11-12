import numpy as np

from lagom.history import BatchEpisode
from lagom.history import BatchSegment


def final_state_from_episode(batch_episode):
    r"""Return the final states of a batch of episodic transitions.
    
    If an episode terminates, then its terminal observation is returned. 
    
    Args:
        batch_episode (BatchEpisode): a batch of episodic transitions. 
        
    Returns
    -------
    out : object
        final states for all episodes from the batch. 
    """
    assert isinstance(batch_episode, BatchEpisode)
    
    obs_shape = batch_episode.env_spec.observation_space.shape
    out = np.zeros((batch_episode.N,) + obs_shape, dtype=np.float32)
    
    for n in range(batch_episode.N):
        if batch_episode.dones[n][-1]:
            out[n, ...] = batch_episode.infos[n][-1]['terminal_observation']
        else:
            out[n, ...] = batch_episode.observations[n][-1]
    
    return out


def final_state_from_segment(batch_segment):
    r"""Return the final states of a batch of rolling segments.
    
    Args:
        batch_segment (BatchSegment): a batch of rolling segments. 
        
    Returns
    -------
    out : object
        final states for all rolling segments from the batch. 
    """
    assert isinstance(batch_segment, BatchSegment)
    
    out = batch_segment.numpy_observations[:, -1, ...]
    for n in range(batch_segment.N):
        if batch_segment.numpy_dones[n, -1]:
            out[n, ...] = batch_segment.infos[n][-1]['terminal_observation']
    
    return out
