from lagom.history import BatchEpisode
from lagom.history import BatchSegment

from lagom.transform import ExpFactorCumSum


def returns_from_episode(batch_episode, gamma):
    assert isinstance(batch_episode, BatchEpisode)
    
    f = ExpFactorCumSum(gamma)
    out = f(batch_episode.numpy_rewards)
    
    return out


def returns_from_segment(batch_segment, gamma):
    assert isinstance(batch_segment, BatchSegment)
    
    f = ExpFactorCumSum(gamma)
    out = f(batch_segment.numpy_rewards, mask=batch_segment.numpy_masks)
    
    return out
