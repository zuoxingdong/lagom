#############TEMP
# TEMP: Import lagom
# Not useful once lagom is installed
import sys
sys.path.append('/home/zuo/Code/lagom/')
##############################


import numpy as np

import unittest

from lagom.core.transform import Clip
from lagom.core.transform import Centralize
from lagom.core.transform import Normalize 
from lagom.core.transform import Standardize
from lagom.core.transform import ExpFactorCumSum
from lagom.core.transform import RunningMeanStd


class TestTransform(unittest.TestCase):
    def test_clip(self):
        clip = Clip()
        
        # Test scalar
        self.assertEqual(clip(x=2, a_min=0, a_max=1), 1)
        self.assertEqual(clip(x=0.5, a_min=0, a_max=1), 0.5)
        self.assertEqual(clip(x=-1, a_min=0, a_max=1), 0)
        
        # Test numpy scalar
        self.assertEqual(clip(x=np.array(2), a_min=0, a_max=1), 1)
        self.assertEqual(clip(x=np.array(0.5), a_min=0, a_max=1), 0.5)
        self.assertEqual(clip(x=np.array(-1), a_min=0, a_max=1), 0)
        
        #
        # Test vector
        #
        def _test_vec(x):
            self.assertTrue(np.alltrue(clip(x=x, a_min=2, a_max=3) == [2, 2, 3, 3]))
        
        # Tuple
        a = (1, 2, 3, 4)
        _test_vec(a)
        
        # List
        b = [1, 2, 3, 4]
        _test_vec(b)
        
        # ndarray
        c = np.array([1, 2, 3, 4])
        _test_vec(c)
        
        #
        # Test exceptions
        #
        # ndarray more than 1-dim is not allowed
        d = np.array([[1, 2, 3, 4]])
        with self.assertRaises(ValueError):
            clip(x=d, a_min=2, a_max=3)
        
    def test_centralize(self):
        centralize = Centralize()
        
        # Test scalar
        self.assertEqual(centralize(x=1), 1)
        self.assertEqual(centralize(x=0), 0)
        self.assertEqual(centralize(x=2), 2)
        
        self.assertEqual(centralize(x=1, mean=1), 1)
        self.assertEqual(centralize(x=0, mean=1), 0)
        self.assertEqual(centralize(x=2, mean=1), 2)
        
        # Test numpy scalar
        self.assertEqual(centralize(x=np.array(1)), 1)
        self.assertEqual(centralize(x=np.array(0)), 0)
        self.assertEqual(centralize(x=np.array(2)), 2)
        
        self.assertEqual(centralize(x=np.array(1), mean=1), 1)
        self.assertEqual(centralize(x=np.array(0), mean=1), 0)
        self.assertEqual(centralize(x=np.array(2), mean=1), 2)
        
        #
        # Test vector
        #
        def _test_vec(x):
            self.assertTrue(np.alltrue(centralize(x=x) == [-1.5, -0.5, 0.5, 1.5]))
            self.assertTrue(np.alltrue(centralize(x=x, mean=1) == [0, 1, 2, 3]))
        
        # Tuple
        a = (1, 2, 3, 4)
        _test_vec(a)
        
        # List
        b = [1, 2, 3, 4]
        _test_vec(b)
        
        # ndarray
        c = np.array([1, 2, 3, 4])
        _test_vec(c)
        
        #
        # Test exceptions
        #
        # ndarray more than 1-dim is not allowed
        d = np.array([[1, 2, 3, 4]])
        with self.assertRaises(ValueError):
            centralize(x=d)
        
    def test_normalize(self):
        normalize = Normalize(eps=1.1920929e-07)
        
        # Test scalar
        self.assertEqual(normalize(x=-1), 0)
        self.assertEqual(normalize(x=0.5), 0.5)
        self.assertEqual(normalize(x=2), 1)
        
        self.assertEqual(normalize(x=-1, min_val=0, max_val=1), 0)
        self.assertEqual(normalize(x=0.5, min_val=0, max_val=1), 0.5)
        self.assertEqual(normalize(x=2, min_val=0, max_val=1), 1)
        
        # Test numpy scalar
        self.assertEqual(normalize(x=np.array(-1)), 0)
        self.assertEqual(normalize(x=np.array(0.5)), 0.5)
        self.assertEqual(normalize(x=np.array(2)), 1)
        
        self.assertEqual(normalize(x=np.array(-1), min_val=0, max_val=1), 0)
        self.assertEqual(normalize(x=np.array(0.5), min_val=0, max_val=1), 0.5)
        self.assertEqual(normalize(x=np.array(2), min_val=0, max_val=1), 1)
        
        #
        # Test vector
        #
        def _test_vec(x):
            self.assertTrue(np.allclose(normalize(x=x), 
                                        [0.        , 0.33333332, 0.66666664, 0.99999996]))
        
            self.assertTrue(np.allclose(normalize(x=x, min_val=0, max_val=1), 
                                        [0.99999988, 1.99999976, 2.99999964, 3.99999952]))
        # Tuple
        a = (1, 2, 3, 4)
        _test_vec(a)
        
        # List
        b = [1, 2, 3, 4]
        _test_vec(b)
        
        # ndarray
        c = np.array([1, 2, 3, 4])
        _test_vec(c)
        
        #
        # Test exceptions
        #
        # ndarray more than 1-dim is not allowed
        d = np.array([[1, 2, 3, 4]])
        with self.assertRaises(ValueError):
            normalize(x=d)
        
    def test_standardize(self):
        standardize = Standardize(eps=1.1920929e-07)
        
        # Test scalar
        self.assertEqual(standardize(x=-1), -1)
        self.assertEqual(standardize(x=0), 0)
        self.assertEqual(standardize(x=1), 1)
        
        self.assertEqual(standardize(x=-1, mean=0, std=1), -1)
        self.assertEqual(standardize(x=0, mean=0, std=1), 0)
        self.assertEqual(standardize(x=1, mean=0, std=1), 1)
        
        # Test numpy scalar
        self.assertEqual(standardize(x=np.array(-1)), -1)
        self.assertEqual(standardize(x=np.array(0)), 0)
        self.assertEqual(standardize(x=np.array(1)), 1)
        
        self.assertEqual(standardize(x=np.array(-1), mean=0, std=1), -1)
        self.assertEqual(standardize(x=np.array(0), mean=0, std=1), 0)
        self.assertEqual(standardize(x=np.array(1), mean=0, std=1), 1)
        
        #
        # Test vector
        #
        def _test_vec(x):
            self.assertTrue(np.allclose(standardize(x=x), 
                                        [-1.34164064, -0.44721355,  0.44721355,  1.34164064]))
        
            self.assertTrue(np.allclose(standardize(x=x, mean=0, std=1), 
                                        [0.99999988, 1.99999976, 2.99999964, 3.99999952]))
        
        # Tuple
        a = (1, 2, 3, 4)
        _test_vec(a)
        
        # List
        b = [1, 2, 3, 4]
        _test_vec(b)
        
        # ndarray
        c = np.array([1, 2, 3, 4])
        _test_vec(c)
        
        #
        # Test exceptions
        #
        # ndarray more than 1-dim is not allowed
        d = np.array([[1, 2, 3, 4]])
        with self.assertRaises(ValueError):
            standardize(x=d)
        
    def test_expfactorcumsum(self):
        expfactorcumsum = ExpFactorCumSum(alpha=0.1)
        
        #
        # Test vector
        #
        def _test_vec(x):
            self.assertTrue(np.allclose(expfactorcumsum(x=x), 
                                        [1.23, 2.3, 3.0]))
        
        # Tuple
        a = (1, 2, 3)
        _test_vec(a)
            
        # List
        b = [1, 2, 3]
        _test_vec(b)
        
        # ndarray
        c = np.array([1, 2, 3])
        _test_vec(c)
        
        #
        # Test exceptions
        #
        # Scalar is not allowed
        with self.assertRaises(AssertionError):
            expfactorcumsum(x=1)
        
        # ndarray more than 1-dim is not allowed
        d = np.array([[1, 2, 3]])
        with self.assertRaises(ValueError):
            expfactorcumsum(x=d)
            
    def test_runningmeanstd(self):
        def _test_moments(runningmeanstd, x):
            self.assertTrue(np.allclose(runningmeanstd.mu, np.mean(x)))
            self.assertTrue(np.allclose(runningmeanstd.sigma, np.std(x)))

        a = [1, 2, 3, 4]

        # Test scalar
        runningmeanstd = RunningMeanStd()
        [runningmeanstd(i) for i in a]
        _test_moments(runningmeanstd=runningmeanstd, x=a)

        # Test vector
        runningmeanstd = RunningMeanStd()
        runningmeanstd(a)
        _test_moments(runningmeanstd=runningmeanstd, x=a)

        # Test n-dim array
        b = np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]])
        runningmeanstd = RunningMeanStd()
        runningmeanstd(b)
        self.assertTrue(np.allclose(runningmeanstd.mu, b.mean(0)))
        self.assertTrue(np.allclose(runningmeanstd.sigma, b.std(0)))


if __name__ == '__main__':
    unittest.main()