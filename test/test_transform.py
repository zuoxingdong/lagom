import numpy as np

import pytest

from lagom.core.transform import Centralize
from lagom.core.transform import Clip
from lagom.core.transform import ExpFactorCumSum
from lagom.core.transform import InterpCurve
from lagom.core.transform import Normalize
from lagom.core.transform import RankTransform
from lagom.core.transform import RunningMeanStd
from lagom.core.transform import Standardize


class TestTransform(object):
    def test_centralize(self): 
        # Not allowed for scalar
        with pytest.raises(AssertionError):
            centralize = Centralize()
            centralize(1.3)

        # single array
        centralize = Centralize()
        out = centralize([1, 2, 3, 4])
        assert out.dtype == np.float32
        assert np.allclose(out, [-1.5, -0.5, 0.5, 1.5])
        assert out.mean() == 0.0

        # batched data
        centralize = Centralize()
        out = centralize([[1, 2], [3, 11]])
        assert out.dtype == np.float32
        assert np.allclose(out, [[-1, -4.5], [1, 4.5]])
        assert np.allclose(out.mean(0), 0.0)

        # from outside mean
        centralize = Centralize()
        mean = np.array([2, 6.5])
        out = centralize([[1, 2], [3, 11]], mean=mean)
        assert out.dtype == np.float32
        assert np.allclose(out, [[-1, -4.5], [1, 4.5]])
        assert np.allclose(out.mean(0), 0.0)
    
    def test_clip(self):
        clip = Clip()

        # Scalar
        y = clip(1.2, 0.5, 1.5)
        assert np.isscalar(y)
        assert np.allclose(y, 1.2)
        del y

        y = clip(0.2, 0.5, 1.5)
        assert np.isscalar(y)
        assert np.allclose(y, 0.5)
        del y

        y = clip(2.3, 0.5, 1.5)
        assert np.isscalar(y)
        assert np.allclose(y, 1.5)
        del y

        # list of values
        y = clip([1, 2, 3], 1.5, 2.5)
        assert y.shape == (3,)
        assert y.dtype == np.float32
        assert np.allclose(y, [1.5, 2, 2.5])
        del y

        y = clip([1, 2, 3], [0, 1, 3.5], [0.5, 2, 8.5])
        assert y.shape == (3, )
        assert y.dtype == np.float32
        assert np.allclose(y, [0.5, 2, 3.5])
        del y

        # ndarray
        y = clip(np.array(2.3), 0.5, 1.5)
        assert np.isscalar(y)
        assert np.allclose(y, 1.5)
        del y

        y = clip(np.array([1, 2, 3]), [0, 1, 3.5], [0.5, 2, 8.5])
        assert y.shape == (3, )
        assert y.dtype == np.float32
        assert np.allclose(y, [0.5, 2, 3.5])
        del y

    def test_expfactorcumsum(self):
        expfactorcumsum = ExpFactorCumSum(alpha=0.1)
        
        #
        # Test vector
        #
        def _test_vec(x):
            # fast code
            assert np.allclose(expfactorcumsum(x=x, mask=None, _fast_code=True), 
                               [1.23, 2.3, 3.0])
            # slow code
            assert np.allclose(expfactorcumsum(x=x, mask=None, _fast_code=False), 
                               [1.23, 2.3, 3.0])
        
        # List
        b = [1, 2, 3]
        _test_vec(b)
        
        # ndarray
        c = np.array([1, 2, 3])
        _test_vec(c)
        
        # 
        # Test with mask
        # 
        def _test_vec_mask(x, mask):
            assert np.allclose(expfactorcumsum(x=x, mask=mask), 
                               [1.23, 2.3, 3.0, 4.56, 5.6, 6.0])
        dones = [False, False, True, False, False, False]
        mask = np.logical_not(dones).astype(int).tolist()
        
        
        # List
        e = [1, 2, 3, 4, 5, 6]
        _test_vec_mask(e, mask)
        
        # ndarray
        f = np.array([1, 2, 3, 4, 5, 6])
        _test_vec_mask(f, mask)
        
        #
        # Test exceptions
        #
        # Scalar is not allowed
        with pytest.raises(AssertionError):
            expfactorcumsum(x=1)
        
        # mask must have same length with input data
        i = [1, 2, 3]
        dones_i = [True, False, True, True]
        mask_i = np.logical_not(dones_i).astype(int).tolist()
        with pytest.raises(AssertionError):
            expfactorcumsum(i, mask_i)
            
        # mask must be binary
        j = [1, 2, 3]
        mask_j = [0, 0.5, 1]
        with pytest.raises(AssertionError):
            expfactorcumsum(j, mask_j)
            
        # boolean mask is not allowed, because it is easy to get bug
        k = [1, 2, 3]
        mask_k = [True, False, False]
        with pytest.raises(AssertionError):
            expfactorcumsum(k, mask_k)

    def test_interp_curve(self):
        interp = InterpCurve()
        
        # Make some inconsistent data
        x1 = [1, 4, 5, 7, 9, 13, 20]
        y1 = [0.1, 0.25, 0.22, 0.53, 0.37, 0.5, 0.55]
        x2 = [2, 4, 6, 7, 9, 11, 15]
        y2 = [0.03, 0.12, 0.4, 0.2, 0.18, 0.32, 0.39]
        
        new_x, (new_y1, new_y2) = interp([x1, x2], [y1, y2], num_point=100)
        
        assert isinstance(new_x, np.ndarray)
        assert isinstance(new_y1, np.ndarray)
        assert isinstance(new_y2, np.ndarray)
        assert new_x.shape == (100,)
        assert new_y1.shape == (100,)
        assert new_y2.shape == (100,)
        assert new_x.min() == 1 and new_x[0] == 1
        assert new_x.max() == 20 and new_x[-1] == 20
        assert new_y1.max() <= 0.6 and new_y2.max() <= 0.6
        
    def test_normalize(self):
        normalize = Normalize(eps=1.1920929e-07)

        # scalar not allowed
        with pytest.raises(AssertionError):
            normalize(-1)

        def _test_vec(x):
            assert np.allclose(normalize(x=x), 
                               [0.        , 0.33333334, 0.6666667 , 1.        ])

            assert np.allclose(normalize(x=x, minimal=0, maximal=1), 
                               [0.9999999, 1.9999998, 2.9999998, 3.9999995])

        # list
        _test_vec([1, 2, 3, 4])

        # ndarray
        _test_vec(np.array([1, 2, 3, 4]))

        # batched data
        out = normalize([[1, 5], [4, 2]])
        assert out.dtype == np.float32
        assert np.allclose(out, [[0, 1], [1, 0]])

        # from outside min and max
        out = normalize([1, 2, 3, 4], minimal=0, maximal=1)
        assert out.dtype == np.float32
        assert np.allclose(out, [0.9999999, 1.9999998, 2.9999998, 3.9999995])
        
    def test_rank_transform(self):
        rank_transform = RankTransform()

        # scalar not allowed
        with pytest.raises(AssertionError):
            rank_transform(3)

        # multidimensional array not allowed
        with pytest.raises(AssertionError):
            rank_transform([[1, 2, 3]])

        # List
        a = [3, 14, 1]
        assert np.allclose(rank_transform(a, centered=False), [1, 2, 0])
        assert np.allclose(rank_transform(a), [0, 0.5, -0.5])

        # ndarray
        b = np.array([3, 14, 1])
        assert np.allclose(rank_transform(b, centered=False), [1, 2, 0])
        assert np.allclose(rank_transform(b), [0, 0.5, -0.5])
        
    def test_runningmeanstd(self):
        def _test_moments(runningmeanstd, x):
            assert runningmeanstd.mu.shape == ()
            assert runningmeanstd.sigma.shape == ()
            assert np.allclose(runningmeanstd.mu, np.mean(x))
            assert np.allclose(runningmeanstd.sigma, np.std(x))

        # different ordering should be the same
        a = [1, 2, 3, 4]
        b = [2, 1, 4, 3]
        c = [4, 3, 2, 1]

        # Scalar
        for item in [a, b, c]:
            runningmeanstd = RunningMeanStd()
            [runningmeanstd(i) for i in item]
            _test_moments(runningmeanstd=runningmeanstd, x=item)

        # Vector
        for item in [a, b, c]:
            runningmeanstd = RunningMeanStd()
            runningmeanstd(item)
            _test_moments(runningmeanstd=runningmeanstd, x=item)

        del a, b, c

        # n-dim array
        b = np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]])
        runningmeanstd = RunningMeanStd()
        runningmeanstd(b)
        assert runningmeanstd.mu.shape == (3,)
        assert runningmeanstd.sigma.shape == (3,)
        assert np.allclose(runningmeanstd.mu, b.mean(0))
        assert np.allclose(runningmeanstd.sigma, b.std(0))

    def test_standardize(self):
        standardize = Standardize(eps=1.1920929e-07)

        # scalar not allowed
        with pytest.raises(AssertionError):
            standardize(1.2)

        def _test_vec(x):
            assert np.allclose(standardize(x=x), 
                               [-1.3416406 , -0.44721353,  0.44721353,  1.3416406 ])
            assert np.allclose(standardize(x=x).mean(), 0.0)
            assert np.allclose(standardize(x=x).std(), 1.0)

            assert np.allclose(standardize(x=x, mean=0, std=1), 
                               [0.9999999, 1.9999998, 2.9999998, 3.9999995])

        # list
        _test_vec([1, 2, 3, 4])

        # ndarray
        _test_vec(np.array([1, 2, 3, 4]))

        # batched data
        out = standardize([[1, 2], [3, 2]])
        assert out.dtype == np.float32
        assert np.allclose(out, [[-1, 0], [1, 0]])
        assert np.allclose(out.mean(0), 0.0)

        # from outside data
        out = standardize([1, 2, 3, 4], mean=0, std=1)
        assert out.dtype == np.float32
        assert np.allclose(out, [1, 2, 3, 4])
