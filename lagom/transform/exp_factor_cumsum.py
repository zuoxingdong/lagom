import numpy as np

from scipy.signal import lfilter

from .base_transform import BaseTransform


class ExpFactorCumSum(BaseTransform):
    r"""Calculate future accumulated sums for each element in a list with an exponential factor. 
    
    Given input data :math:`[x_1, ..., x_n]` and exponential factor :math:`\alpha\in [0, 1]`, it returns
    an array :math:`y` with the same length and each element is calculated as following
    
    .. math::
        y_i = x_i + \alpha*x_{i+1} + \alpha^2*x_{i+2} + \dots + \alpha^{n-i-1}*x_{n-1} + \alpha^{n-i}*x_{n}
        
    .. note::
    
        We provided a fast and a slow implementations. For fast implementation, it uses
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
    
        >>> f = ExpFactorCumSum(0.1)
        >>> f([1, 2, 3, 4])
        [1.234, 2.34, 3.4, 4]
        
        >>> f([1, 2, 3, 4], mask=[1, 0, 1, 1], _fast_code=True)
        [1.2, 2.0, 3.4, 4]

    """
    def __init__(self, alpha):
        r"""Initialize transformation. 
        
        Args:
            alpha (float): exponential factor between zero and one. 
        """
        self.alpha = alpha
        
    def __call__(self, x, mask=None):
        r"""Calculate future accumulated sums with exponential factor. 
        
        An optional binary mask could be used. 
        Intuitively, the computation will restart for each occurrence
        of zero in the mask. If ``None``, then default mask is ones everywhere.
        
        Args:
            x (list): input data
            mask (list, optional): binary mask (zero or one) corresponds to each element. 
                Default: ``None``
            _fast_code (bool, optinal): if ``True``, then use fast implementation based 
                on build-in function ``accumulate()``. Otherwise, use vanilla implementation.
                Default: ``True``

        Returns
        -------
        out : list
            calculated data
        """
        assert not np.isscalar(x), 'does not support scalar value !'
        x = self.to_numpy(x, np.float32)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        
        if mask is not None:
            assert np.asarray(mask).dtype != np.bool
            assert np.array_equal(mask, np.array(mask).astype(bool))
            
            mask = self.to_numpy(mask, np.float32)
            if mask.ndim == 1:
                mask = np.expand_dims(mask, 0)
            assert mask.ndim == 2
            assert mask.shape == x.shape
        
        if mask is None:
            return lfilter([1], [1, -self.alpha], x[:, ::-1], axis=1)[:, ::-1]
        else:
            N, T = x.shape
            out = np.zeros_like(x, dtype=np.float32)
            cumsum = np.zeros(N, dtype=np.float32)

            for t in reversed(range(T)):
                cumsum = x[:, t] + self.alpha*cumsum*mask[:, t]
                out[:, t] = cumsum

            return out
