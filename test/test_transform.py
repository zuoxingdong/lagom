import numpy as np

import pytest

from lagom.transform import interp_curves
from lagom.transform import geometric_cumsum
from lagom.transform import explained_variance
from lagom.transform import LinearSchedule
from lagom.transform import rank_transform
from lagom.transform import PolyakAverage
from lagom.transform import RunningMeanVar
from lagom.transform import smooth_filter


def test_interp_curves():
    # Make some inconsistent data
    x1 = [4, 5, 7, 13, 20]
    y1 = [0.25, 0.22, 0.53, 0.37, 0.55]
    x2 = [2, 4, 6, 7, 9, 11, 15]
    y2 = [0.03, 0.12, 0.4, 0.2, 0.18, 0.32, 0.39]
    new_x, new_y = interp_curves([x1, x2], [y1, y2], num_point=100)
    
    assert isinstance(new_x, list)
    assert isinstance(new_y, list)
    assert len(new_x[0]) == 100
    assert len(new_y[0]) == 100
    assert len(new_x[1]) == 100
    assert len(new_y[1]) == 100
    assert new_x[0] == new_x[1]
    assert min(new_x[0]) == 2 and min(new_x[1]) == 2
    assert max(new_x[0]) <= 20 and max(new_x[1]) <= 20
    assert min(new_y[0]) > 0 and min(new_y[1]) > 0
    assert max(new_y[0]) <= 0.6 and max(new_y[1]) <= 0.6


def test_geometric_cumsum():
    assert np.allclose(geometric_cumsum(0.1, [1, 2, 3]), [1.23, 2.3, 3])
    assert np.allclose(geometric_cumsum(0.1, [[1, 2, 3, 4], [5, 6, 7, 8]]), 
                       [[1.234, 2.34, 3.4, 4], [5.678, 6.78, 7.8, 8]])


def test_explained_variance():
    assert np.isclose(explained_variance(y_true=[3, -0.5, 2, 7], y_pred=[2.5, 0.0, 2, 8]), 0.9571734666824341)
    assert np.isclose(explained_variance(y_true=[[3, -0.5, 2, 7]], y_pred=[[2.5, 0.0, 2, 8]]), 0.9571734666824341)
    assert np.isclose(explained_variance(y_true=[[0.5, 1], [-1, 1], [7, -6]], y_pred=[[0, 2], [-1, 2], [8, -5]]), 
                      0.9838709533214569)
    assert np.isclose(explained_variance(y_true=[[0.5, 1], [-1, 10], [7, -6]], y_pred=[[0, 2], [-1, 0.00005], [8, -5]]), 
                      0.6704022586345673)


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
    assert all([scheduler(i) == scheduler.get_current() for i in range(10)])

    # increasing: with warmup start
    scheduler = LinearSchedule(initial=0.5, final=2.0, N=3, start=2)
    assert all([scheduler(i) == 0.5] for i in [0, 1, 2])
    assert scheduler(3) == 1.0
    assert scheduler(4) == 1.5
    assert scheduler(5) == 2.0
    assert all([scheduler(i) == 2.0 for i in [6, 7, 8]])
    assert all([scheduler(i) == scheduler.get_current() for i in range(10)])

    # decreasing: without warmup start
    scheduler = LinearSchedule(initial=1.0, final=0.1, N=3, start=0)
    assert scheduler(0) == 1.0
    assert scheduler(1) == 0.7
    assert scheduler(2) == 0.4
    assert scheduler(3) == 0.1
    assert all([scheduler(i) == 0.1 for i in [4, 5, 6]])
    assert all([scheduler(i) == scheduler.get_current() for i in range(10)])

    # decreasing: with warmup start
    scheduler = LinearSchedule(initial=1.0, final=0.1, N=3, start=2)
    assert all([scheduler(i) for i in [0, 1, 2]])
    assert scheduler(3) == 0.7
    assert scheduler(4) == 0.4
    assert scheduler(5) == 0.1
    assert all([scheduler(i) == 0.1 for i in [6, 7, 8]])
    assert all([scheduler(i) == scheduler.get_current() for i in range(10)])


def test_rank_transform():
    with pytest.raises(AssertionError):
        rank_transform(3)
    with pytest.raises(AssertionError):
        rank_transform([[1, 2, 3]])
    
    assert np.allclose(rank_transform([3, 14, 1], centered=True), [0, 0.5, -0.5])
    assert np.allclose(rank_transform([3, 14, 1], centered=False), [1, 2, 0])


def test_polyak_average():
    with pytest.raises(AssertionError):
        PolyakAverage(alpha=-1.0)
    with pytest.raises(AssertionError):
        PolyakAverage(alpha=1.2)
        
    f = PolyakAverage(alpha=0.1)
    x = f(0.5)
    assert np.allclose(x, 0.5)
    x = f(1.5)
    assert np.allclose(x, 1.4)
    assert np.allclose(x, f.get_current())
    
    f = PolyakAverage(alpha=0.1)
    x = f(1.0)
    assert np.allclose(x, 1.0)
    x = f(2.0)
    assert np.allclose(x, 1.9)
    assert np.allclose(x, f.get_current())
    
    f = PolyakAverage(alpha=0.1)
    x = f([0.5, 1.0])
    assert np.allclose(x, [0.5, 1.0])
    x = f([1.5, 2.0])
    assert np.allclose(x, [1.4, 1.9])
    assert np.allclose(x, f.get_current())


def test_running_mean_var():
    with pytest.raises(AssertionError):
        f = RunningMeanVar(shape=())
        f(0.5)
    with pytest.raises(AssertionError):
        f = RunningMeanVar(shape=(1,))
        f([0.5])
    
    x = np.random.randn(1000)
    xs = np.array_split(x, [1, 200, 500, 600, 900, 950])
    assert np.allclose(np.concatenate(xs), x)
    f = RunningMeanVar(shape=())
    for x_part in xs:
        f(x_part)
    assert np.allclose(f.mean, x.mean())
    assert np.allclose(f.var, x.var())
    assert np.allclose(np.sqrt(f.var + 1e-8), x.std())
    assert f.n == 1000

    x = np.random.randn(1000, 32, 3)
    xs = np.array_split(x, [1, 200, 500, 600, 900, 950])
    assert np.allclose(np.concatenate(xs), x)
    f = RunningMeanVar(shape=(32, 3))
    for x_part in xs:
        f(x_part)
    assert np.allclose(f.mean, x.mean(0))
    assert np.allclose(f.var, x.var(0))
    assert np.allclose(np.sqrt(f.var + 1e-8), x.std(0))
    assert f.n == 1000


def test_smooth_filter():
    with pytest.raises(AssertionError):
        smooth_filter([[1, 2, 3]], window_length=3, polyorder=2)
    
    x = np.linspace(0, 4*2*np.pi, num=100)
    y = x*(np.sin(x) + np.random.random(100)*4)
    out = smooth_filter(y, window_length=31, polyorder=10)
    assert out.shape == (100,)
