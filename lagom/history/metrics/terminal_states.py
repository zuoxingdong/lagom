import numpy as np

from lagom.history import BatchEpisode
from lagom.history import BatchSegment


def terminal_state_from_episode(batch_episode):
    r"""Return the terminal states of a batch of episodic transitions, if available. 
    
    If no terminal state exists, then an ``None`` is returned. 
    
    Args:
        batch_episode (BatchEpisode): a batch of episodic transitions. 
        
    Returns
    -------
    out : object
        terminal states for all episodes from the batch. 
    """
    assert isinstance(batch_episode, BatchEpisode)
    
    final_dones = [dones[-1] for dones in batch_episode.dones]
    N = final_dones.count(True)
    
    if N == 0:
        return None
    else:
        obs_shape = batch_episode.env_spec.observation_space.shape
        out = np.zeros((N,) + obs_shape, dtype=np.float32)
        indices = np.where(final_dones)[0]
        for i, idx in enumerate(indices):
            out[i, ...] = batch_episode.infos[idx][-1]['terminal_observation']
        return out
    
    
def terminal_state_from_segment(batch_segment):
    r"""Return the terminal states of a batch of rolling segments if available. 
    
    If no terminal state exists, then an ``None`` is returned. 
    
    Args:
        batch_segment (BatchSegment): a batch of rolling segments. 
        
    Returns
    -------
    out : object
        terminal states for all rolling segments from the batch. 
    """
    assert isinstance(batch_segment, BatchSegment)
    
    if np.allclose(batch_segment.numpy_dones, False):
        return None
    else:
        indices = np.vstack(np.where(batch_segment.numpy_dones == True)).T
        obs_shape = batch_segment.env_spec.observation_space.shape
        out = np.zeros((len(indices),) + obs_shape, dtype=np.float32)
        for i, (n, t) in enumerate(indices):
            out[i, ...] = batch_segment.infos[n][t]['terminal_observation']
        return out
