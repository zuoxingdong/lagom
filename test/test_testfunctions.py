import numpy as np

import pytest

from lagom.core.es.test_functions import Rastrigin
from lagom.core.es.test_functions import Sphere
from lagom.core.es.test_functions import StyblinskiTang
from lagom.core.es.test_functions import HolderTable


class TestTestFunctions(object):
    def test_rastrigin(self):
        rastrigin = Rastrigin()
        
        assert rastrigin([0]) == 0.0
        assert rastrigin([0, 0]) == 0.0
        assert rastrigin([0]*100) == 0.0
        
        assert np.allclose(rastrigin([5.12]), 28.924713725785896)
        
    def test_sphere(self):
        sphere = Sphere()
        
        assert sphere([0]) == 0.0
        assert sphere([2]) == 4.0
        
        assert sphere([1, 2, 3]) == 14.0
        
    def test_styblinski_tang(self):
        styblinski_tang = StyblinskiTang()
        
        assert styblinski_tang([0]) == 0.0
        assert styblinski_tang([1]) == -5.0
        
        assert styblinski_tang([1, 2]) == -24.0
        
    def test_holder_table(self):
        holder_table = HolderTable()
        
        assert holder_table([0, 0]) == 0.0
        
        assert np.allclose(holder_table([8.05502, 9.66459]), -19.2085)
        assert np.allclose(holder_table([-8.05502, 9.66459]), -19.2085)
        assert np.allclose(holder_table([8.05502, -9.66459]), -19.2085)
        assert np.allclose(holder_table([-8.05502, -9.66459]), -19.2085)