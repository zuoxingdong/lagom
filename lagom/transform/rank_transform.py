import numpy as np


def rank_transform(x, centered=True):
    r"""Rank transformation of a vector of values. The rank has the same dimensionality as the vector.
    Each element in the rank indicates the index of the ascendingly sorted input.
    i.e. ``ranks[i] = k``, it means i-th element in the input is :math:`k`-th smallest value. 
    
    Rank transformation reduce sensitivity to outliers, e.g. in OpenAI ES, gradient computation
    involves fitness values in the population, if there are outliers (too large fitness), it affects
    the gradient too much. 
    
    Note that a centered rank transformation to the range [-0.5, 0.5] is supported by an option. 
    
    Example:
    
        >>> rank_transform([3, 14, 1], centered=True)
        array([ 0. ,  0.5, -0.5])
        
        >>> rank_transform([3, 14, 1], centered=False)
        array([1, 2, 0])
        
    Args:
        x (list/ndarray): a vector of values.
        centered (bool, optional): if ``True``, then centered the rank transformation 
            to :math:`[-0.5, 0.5]`. Defualt: ``True``

    Returns:
        ndarray: an numpy array of ranks of input data
    """
    x = np.asarray(x)
    assert x.ndim == 1, 'must be one dimensional, i.e. a vector of scalar values'
    
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    if centered:
        ranks = ranks/(ranks.size - 1) - 0.5
    return ranks
