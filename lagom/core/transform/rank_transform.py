import numpy as np

from .base_transform import BaseTransform


class RankTransform(BaseTransform):
    """
    Rank transformation of the input vector. The rank has the same dimensionality as input vector.
    Each element in the rank indicates the index of the ascendingly sorted input.
    i.e. ranks[i] = k, it means i-th element in the input is k-th smallest value. 
    
    Rank transformation reduce sensitivity to outliers, e.g. in OpenAI ES, gradient computation
    involves fitness values in the population, if there are outliers (too large fitness), it affects
    the gradient too much. 
    
    Note that a centered rank transformation to the range [-0.5, 0.5] is supported by an option. 
    """
    def __call__(self, x, centered=True):
        """
        Rank transformation of the input vector. 
        
        Args:
            x (list/ndarray): Input data. Note that it should be only 1-dim vector
            centered (bool): If True, then centered the rank transformation to [-0.5, 0.5]

        Returns:
            ranks (ndarray): Ranks of input data

        Examples:
            >>> x = [3, 14, 1]
            >>> rank_transform = RankTransform()
            >>> rank_transform(x, centered=True)
        """
        # Convert input to ndarray
        x = self.make_input(x)
        assert x.ndim == 1, 'Only 1-dim vector is supported. Received a scalar value. '
        
        # Compute ranks
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        
        # Centered ranks
        if centered:
            ranks = ranks/(ranks.size - 1) - 0.5
            
        return ranks.astype(np.float32)
