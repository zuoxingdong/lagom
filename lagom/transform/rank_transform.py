import numpy as np

from .base_transform import BaseTransform


class RankTransform(BaseTransform):
    r"""Rank transformation of a vector of values. The rank has the same dimensionality as the vector.
    Each element in the rank indicates the index of the ascendingly sorted input.
    i.e. ``ranks[i] = k``, it means i-th element in the input is :math:`k`-th smallest value. 
    
    Rank transformation reduce sensitivity to outliers, e.g. in OpenAI ES, gradient computation
    involves fitness values in the population, if there are outliers (too large fitness), it affects
    the gradient too much. 
    
    Note that a centered rank transformation to the range [-0.5, 0.5] is supported by an option. 
    
    Example::
    
        >>> ranks = RankTransform()
        >>> ranks([3, 14, 1], centered=True)
        array([ 0. ,  0.5, -0.5])
        
        >>> ranks([3, 14, 1], centered=False)
        array([1, 2, 0])
    
    """
    def __call__(self, x, centered=True):
        r"""Rank transformation of the vector. 
        
        Args:
            x (list/ndarray): a vector of values.
            centered (bool, optional): if ``True``, then centered the rank transformation 
                to :math:`[-0.5, 0.5]`. Defualt: ``True``

        Returns
        -------
        ranks : ndarray
            ranks of input data
        """
        assert not np.isscalar(x), 'does not support scalar value !'
        
        x = self.to_numpy(x, np.float32)
        
        assert x.ndim == 1, 'must be one dimensional, i.e. a vector of scalar values'
        
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        
        if centered:
            ranks = ranks/(ranks.size - 1) - 0.5
        
        return ranks
