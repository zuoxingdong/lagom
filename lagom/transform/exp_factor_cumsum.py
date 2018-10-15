import numpy as np

from itertools import accumulate

from .base_transform import BaseTransform


class ExpFactorCumSum(BaseTransform):
    r"""Calculate future accumulated sums with exponential factor for each element in a list. 
    
    Given input :math:`[x_1, ..., x_n]` and factor :math:`\alpha\in [0, 1]`, then it returns
    an array :math:`y` with same length and each element is calculated as following
    
    .. math::
        y_i = x_i + \alpha*x_{i+1} + \alpha^2*x_{i+2} + \dots + \alpha^{n-i-1}*x_{n-1} + \alpha^{n-i}*x_{n}
        
    .. note::
    
        We have provided two implementations, fast and slow. For fast implementation, we use
        Python build-in function ``itertools.accumulate`` and the slow one is implementated
        by using for looping. According to the benchmarks, the amount of speedup with respect
        to the sequence length for the fast implementation over the slow one is shown as following:
            
            * Length 100: :math:`\approx 1%` faster
            * Length 1000: :math:`\approx 33%` faster
            * Length 2000: :math:`\approx 69%` faster
            
        Because this function is commonly used for calculating discounted returns for every time
        step in an episode, and often an episode has length of a few hundreds only. There might
        not be a significant speedup. 
        
    .. warning::
    
        Currently, the batched calculation is not supported !
    
    Example::
    
        >>> f = ExpFactorCumSum(alpha=0.1)
        >>> f([1, 2, 3, 4], _fast_code=True)
        [1.234, 2.34, 3.4, 4]
        
        >>> f([1, 2, 3, 4], mask=[1, 0, 1, 1], _fast_code=True)
        [1.2, 2.0, 3.4, 4]

    """
    def __init__(self, alpha):
        r"""Initialize transoformation. 
        
        Args:
            alpha (float): exponential factor between zero and one. 
        """
        self.alpha = alpha
        
    def __call__(self, x, mask=None, _fast_code=True):
        r"""Calculate future accumulated sums with exponential factor. 
        
        An optional binary mask could be used. 
        Intuitively, the computation will restart for each occurrence
        of zero in the mask. If ``None``, then default mask is ones everywhere.
        
        Args:
            x (list): input data
            mask (list, optional): binary mask (zero or one) corresponds to each data item. 
                Default: ``None``
            _fast_code (bool, optinal): if ``True``, then use fast implementation based 
                on build-in function ``accumulate()``. Otherwise, use vanilla implementation.
                Default: ``True``

        Returns
        -------
        out : list
            calculated data
        """
        assert not np.isscalar(x), 'not allowed to use scalar value !'
        if isinstance(x, np.ndarray):
            x = x.tolist()
        assert isinstance(x, list), f'expected list, got {type(x)}'
        
        # make default mask
        if mask is None:
            mask = [1.0]*len(x)
        else:  # sanity check for non-boolean and must be binaries
            msg = 'mask should use binary values as zero or one. Not bool, because it is prone to bug '
            msg += 'e.g. gym done=True, but we want to use mask value zero instead to compute returns. '
            assert np.asarray(mask).dtype != 'bool', msg
        
            msg = 'The mask must be binary, i.e. either 0 or 1. '
            assert np.array_equal(mask, np.array(mask).astype(bool))
        assert len(x) == len(mask), f'mask must be same length with data, {len(mask)} != {len(x)}'
        
        # choose fast or slow implementation
        if _fast_code:
            return self._fast(x, mask)
        else:  # use slow vanilla implementation
            return self._slow(x, mask)

    def _slow(self, x, masks):
        cumsum = 0.0
        out = []

        # iterate over items reversely
        for value, mask in zip(x[::-1], masks[::-1]):
            cumsum = value + self.alpha*cumsum*mask  # recursive update
            out.insert(0, cumsum)  # insert to the front

        return out
    
    def _fast(self, x, masks):
        # zip the input value and masks with reverse ordering
        D = list(zip(x[::-1], masks[::-1]))
        # replace first element with its x value, drop out the mask item
        # because accumulating sum, the mask in the last is not useful to compute total value
        D[0] = D[0][0]
        # compute it by using accumulate()
        out = accumulate(D, lambda total, element: element[0] + self.alpha*total*element[1])
        out = list(out)[::-1]
        
        return out
