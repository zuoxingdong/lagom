import numpy as np

import pytest

from lagom.transform import Centralize
from lagom.transform import Clip
from lagom.transform import ExpFactorCumSum
from lagom.transform import ExplainedVariance
from lagom.transform import InterpCurve
from lagom.transform import LinearSchedule
from lagom.transform import Normalize
from lagom.transform import RankTransform
from lagom.transform import RunningMeanStd
from lagom.transform import SmoothFilter
from lagom.transform import Standardize


def check_dtype(out):
    if np.isscalar(out):
        assert isinstance(out, np.float32)
    else:
        assert out.dtype == np.float32
    

def test_centralize():
    centralize = Centralize()
    
    with pytest.raises(AssertionError):
        centralize(1.2, 0)

    out = centralize([1, 2, 3, 4], 0)
    assert np.allclose(out, [-1.5, -0.5, 0.5, 1.5])
    assert out.mean() == 0.0
    check_dtype(out)

    out = centralize([[1, 3], [2, 4], [3, 5]], 0)
    assert np.allclose(out, [[-1, -1], [0, 0], [1, 1]])
    assert np.allclose(out.mean(0), 0.0)
    check_dtype(out)

    out = centralize([[1, 3], [2, 4], [3, 5]], 1)
    assert np.allclose(out, [[-1, 1], [-1, 1], [-1, 1]])
    assert np.allclose(out.mean(1), 0.0)
    check_dtype(out)

    mean = [0.1, 0.2, 0.3]
    out = centralize([1, 2, 3], 0, mean=mean)
    assert np.allclose(out, [0.9, 1.8, 2.7])
    check_dtype(out)

    mean = [0.1, 0.2]
    out = centralize([[1, 3], [2, 4], [3, 5]], 0, mean=mean)
    assert np.allclose(out, [[0.9, 2.8], [1.9, 3.8], [2.9, 4.8]])
    check_dtype(out)
    
    
def test_clip():
    clip = Clip()

    out = clip(1.2, 0.5, 1.5)
    assert np.isscalar(out) and np.allclose(out, 1.2)
    check_dtype(out)

    out = clip(0.2, 0.5, 1.5)
    assert np.isscalar(out) and np.allclose(out, 0.5)
    check_dtype(out)

    out = clip(2.3, 0.5, 1.5)
    assert np.isscalar(out) and np.allclose(out, 1.5)
    check_dtype(out)

    out = clip([1, 2, 3], 1.5, 2.5)
    assert out.shape == (3,) and np.allclose(out, [1.5, 2, 2.5])
    check_dtype(out)

    out = clip([1, 2, 3], [0, 1, 3.5], [0.5, 2, 9])
    assert out.shape == (3,) and np.allclose(out, [0.5, 2, 3.5])
    check_dtype(out)
    
    
def test_expfactorcumsum():
    f = ExpFactorCumSum(0.1)

    with pytest.raises(AssertionError):
        f(1.5)

    assert np.allclose(f([1, 2, 3], _fast_code=True), [1.23, 2.3, 3])
    assert np.allclose(f([1, 2, 3], _fast_code=False), [1.23, 2.3, 3])

    x = [1, 2, 3, 4, 5, 6]
    dones = [False, False, True, False, False, False]
    mask = np.logical_not(dones).astype(int).tolist()

    assert np.allclose(f(x, mask=mask), [1.23, 2.3, 3, 4.56, 5.6, 6])

    with pytest.raises(AssertionError):
        f(x, mask=dones)

    with pytest.raises(AssertionError):
        f(x, mask=[1, 0, 1])

    with pytest.raises(AssertionError):
        f(x, mask=[1, 1, 0.5, 1, 1, 1])


def test_explained_variance():
    f = ExplainedVariance()

    with pytest.raises(AssertionError):
        f(1, 1)
    with pytest.raises(AssertionError):
        f([1, 2, 3], 2)

    ev = f([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
    assert np.isscalar(ev) and np.isclose(ev, 0.9571734666824341)
    check_dtype(ev)

    ev = f([[0.5, 1], [-1, 1], [7, -6]], [[0, 2], [-1, 2], [8, -5]])
    assert np.isscalar(ev) and np.isclose(ev, 0.9838709533214569)
    check_dtype(ev)

    ev = f([[0.5, 1], [-1, 10], [7, -6]], [[0, 2], [-1, 0.00005], [8, -5]])
    assert np.isscalar(ev) and np.isclose(ev, 0.6704022586345673)
    check_dtype(ev)
    

def test_interp_curve():
    interp = InterpCurve()

    # Make some inconsistent data
    x1 = [1, 4, 5, 7, 9, 13, 20]
    y1 = [0.1, 0.25, 0.22, 0.53, 0.37, 0.5, 0.55]
    x2 = [2, 4, 6, 7, 9, 11, 15]
    y2 = [0.03, 0.12, 0.4, 0.2, 0.18, 0.32, 0.39]

    new_x, (new_y1, new_y2) = interp([x1, x2], [y1, y2], num_point=100)

    check_dtype(new_x)
    check_dtype(new_y1)
    check_dtype(new_y2)
    assert isinstance(new_x, np.ndarray)
    assert isinstance(new_y1, np.ndarray) and isinstance(new_y2, np.ndarray)
    assert new_x.shape == (100,)
    assert new_y1.shape == (100,) and new_y2.shape == (100,)
    assert new_x.min() == 1 and new_x[0] == 1
    assert new_x.max() == 20 and new_x[-1] == 20
    assert new_y1.max() <= 0.6 and new_y2.max() <= 0.6
    
    
def test_linear_schedule():
    with pytest.raises(AssertionError):
        LinearSchedule(1.0, 0.1, 0, 0)
    with pytest.raises(AssertionError):
        LinearSchedule(1.0, 0.1, -1, 0)
    with pytest.raises(AssertionError):
        LinearSchedule(1.0, 0.1, 10, -1)
    with pytest.raises(AssertionError):
        LinearSchedule(1.0, 0.1, 10, 0)(-1)

    # increasing: without warmup start
    scheduler = LinearSchedule(initial=0.5, final=2.0, N=3, start=0)
    assert scheduler(0) == 0.5
    assert scheduler(1) == 1.0
    assert scheduler(2) == 1.5
    assert scheduler(3) == 2.0
    assert all([scheduler(i) == 2.0] for i in [4, 5, 6, 7, 8])

    # increasing: with warmup start
    scheduler = LinearSchedule(initial=0.5, final=2.0, N=3, start=2)
    assert all([scheduler(i) == 0.5] for i in [0, 1, 2])
    assert scheduler(3) == 1.0
    assert scheduler(4) == 1.5
    assert scheduler(5) == 2.0
    assert all([scheduler(i) == 2.0 for i in [6, 7, 8]])

    # decreasing: without warmup start
    scheduler = LinearSchedule(initial=1.0, final=0.1, N=3, start=0)
    assert scheduler(0) == 1.0
    assert scheduler(1) == 0.7
    assert scheduler(2) == 0.4
    assert scheduler(3) == 0.1
    assert all([scheduler(i) == 0.1 for i in [4, 5, 6]])

    # decreasing: with warmup start
    scheduler = LinearSchedule(initial=1.0, final=0.1, N=3, start=2)
    assert all([scheduler(i) for i in [0, 1, 2]])
    assert scheduler(3) == 0.7
    assert scheduler(4) == 0.4
    assert scheduler(5) == 0.1
    assert all([scheduler(i) == 0.1 for i in [6, 7, 8]])
    
    
def test_normalize():
    normalize = Normalize(eps=1.1920929e-07)

    with pytest.raises(AssertionError):
        normalize(-1, 0)

    out = normalize([1, 2, 3, 4], 0)
    assert np.allclose(out, [0., 0.33333334, 0.6666667, 1.])
    check_dtype(out)

    out = normalize([1, 2, 3, 4], 0, minimal=0, maximal=1)
    assert np.allclose(out, [0.9999999, 1.9999998, 2.9999998, 3.9999995])
    check_dtype(out)

    out = normalize([[1, 5, 8], [2, 4, 6]], 0)
    assert np.allclose(out, [[0, 0.999999, 1], [0.999999, 0, 0]])
    check_dtype(out)

    out = normalize([[1, 5, 8], [2, 4, 6]], 1)
    assert np.allclose(out, [[0, 0.5714286, 1], [0, 0.5, 1]])
    check_dtype(out)

    out = normalize([1, 2, 3, 4], 0, minimal=1, maximal=3)
    assert np.allclose(out, [0., 0.49999994, 0.9999999, 1.4999999]) 
    check_dtype(out)
    

def test_rank_transform():
    f = RankTransform()

    with pytest.raises(AssertionError):
        f(3)
    with pytest.raises(AssertionError):
        f([[1, 2, 3]])
    
    x = [3, 14, 1]
    out = f(x, centered=False)
    assert np.allclose(out, [1, 2, 0])
    
    out = f(x)
    assert np.allclose(out, [0, 0.5, -0.5])

    
def test_runningmeanstd():
    def check(runningmeanstd, x):
        assert runningmeanstd.mu.shape == ()
        assert runningmeanstd.sigma.shape == ()
        assert np.allclose(runningmeanstd.mu, np.mean(x))
        assert np.allclose(runningmeanstd.sigma, np.std(x))

    a = [1, 2, 3, 4]
    b = [2, 1, 4, 3]
    c = [4, 3, 2, 1]

    for item in [a, b, c]:
        runningmeanstd = RunningMeanStd()
        for i in item:
            runningmeanstd(i)
        check(runningmeanstd, item)

    for item in [a, b, c]:
        runningmeanstd = RunningMeanStd()
        runningmeanstd(item)
        check(runningmeanstd, item)

    del a, b, c

    b = np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]])
    runningmeanstd = RunningMeanStd()
    runningmeanstd(b)
    assert runningmeanstd.mu.shape == (3,)
    assert runningmeanstd.sigma.shape == (3,)
    assert np.allclose(runningmeanstd.mu, b.mean(0))
    assert np.allclose(runningmeanstd.sigma, b.std(0))    


def test_smooth_filter():
    f = SmoothFilter()

    with pytest.raises(AssertionError):
        f(1)
    with pytest.raises(AssertionError):
        f([1, 2, 3], window_length=3)
    with pytest.raises(AssertionError):
        f([1, 2, 3], polyorder=2)

    x = np.linspace(0, 4*2*np.pi, num=100)
    y = x*(np.sin(x) + np.random.random(100)*4)
    out = f(y, window_length=31, polyorder=10)
    assert out.shape == (100,)
    assert out.dtype == np.float32

      
def test_standardize():
    standardize = Standardize(eps=1.1920929e-07)

    with pytest.raises(AssertionError):
        standardize(1.2, 0)
        
    out = standardize([1, 2, 3, 4], 0)
    assert np.allclose(out, [-1.3416406, -0.44721353, 0.44721353, 1.3416406])
    assert np.isclose(out.mean(), 0.0)
    assert np.isclose(out.std(), 1.0)
    check_dtype(out)
    
    out = standardize([1, 2, 3, 4], 0, mean=1, std=2)
    assert np.allclose(out, [0, 0.5, 1, 1.5])
    check_dtype(out)

    out = standardize([[5, 3, 12], [-40, 2, 16], [17, 8, -2]], 0)
    assert np.allclose(out, [[0.44832653, -0.5080005, 0.43193418],
                             [-1.3857366, -0.88900083, 0.9502552],
                             [0.93741, 1.3970011, -1.3821895]])
    assert np.allclose(out.mean(0), 0.0, atol=1e-7)
    assert not np.allclose(out.mean(1), 0.0, atol=1e-7)
    assert np.allclose(out.std(0), 1.0)
    assert not np.allclose(out.std(1), 1.0)
    check_dtype(out)
        
    out = standardize([[5, 3, 12], [-40, 2, 16], [17, 8, -2]], 1)
    assert np.allclose(out, [[-0.43193415, -0.95025516, 1.3821894],
                             [-1.372813, 0.39223227, 0.9805807],
                             [1.2027031, 0.0429537, -1.2456566]])
    assert np.allclose(out.mean(1), 0.0, atol=1e-7)
    assert not np.allclose(out.mean(0), 0.0, atol=1e-7)
    assert np.allclose(out.std(1), 1.0)
    assert not np.allclose(out.std(0), 1.0)
    check_dtype(out)

    out = standardize([[5, 3, 12], [-40, 2, 16], [17, 8, -2]], 0, mean=[1, 2, 3], std=[4, 5, 6])
    assert np.allclose(out, [[1., 0.2, 1.5],
                             [-10.25, 0., 2.1666667],
                             [4., 1.2, -0.8333333]])
    check_dtype(out)
