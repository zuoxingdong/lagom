import numpy as np

import pytest

from lagom.es import BaseES
from lagom.es import BaseESMaster
from lagom.es import BaseESWorker

from lagom.es import rastrigin
from lagom.es import sphere
from lagom.es import holder_table
from lagom.es import styblinski_tang


class ES(BaseES):
    def __init__(self):
        self.count = 0
        self.x = [1, 2, 3]
        self.best_f = 0.0
        self.popsize = 3
        
    def ask(self):
        solutions = self.x
        
        self.count += 1
        self.x = [i + 1 for i in self.x]
        
        return solutions
        
    def tell(self, solutions, function_values):
        self.best_f = max(function_values)
        
    @property
    def result(self):
        return {'best_f': self.best_f}
    
    
class ESMaster(BaseESMaster):
    def make_es(self, config):
        es = ES()
        
        return es
    
    def process_es_result(self, result):
        assert 'best_f' in result
        if self.generation == 0:
            assert result['best_f'] == 9
        elif self.generation == 1:
            assert result['best_f'] == 16
        elif self.generation == 2:
            assert result['best_f'] == 25
        elif self.generation == 3:
            assert result['best_f'] == 36


class ESWorker(BaseESWorker):
    def prepare(self):
        self.prepared = True
        
    def f(self, config, solution):
        assert self.prepared
        
        return solution**2

    
def test_es():
    es = ES()
    assert es.count == 0
    assert es.x == [1, 2, 3]
    assert es.best_f == 0.0
    assert es.popsize == 3
    
    solutions = es.ask()
    es.tell(solutions, [1.2, 3.2, 1.3])
    assert solutions == [1, 2, 3]
    assert es.count == 1
    assert es.x == [2, 3, 4]
    assert es.best_f == 3.2
    
    result = es.result
    assert isinstance(result, dict) and 'best_f' in result and result['best_f'] == 3.2
    
    
def test_es_master_worker():
    es = ESMaster({'train.num_iteration': 4}, ESWorker)
    es()


def test_rastrigin():
    assert rastrigin([0]) == 0.0
    assert rastrigin([0, 0]) == 0.0
    assert rastrigin([0]*100) == 0.0
    
    assert np.allclose(rastrigin([5.12]), 28.924713725785896)


def test_sphere():
    assert sphere([0]) == 0.0
    assert sphere([2]) == 4.0
        
    assert sphere([1, 2, 3]) == 14.0


def test_holder_table():
    assert holder_table([0, 0]) == 0.0
        
    assert np.allclose(holder_table([8.05502, 9.66459]), -19.2085)
    assert np.allclose(holder_table([-8.05502, 9.66459]), -19.2085)
    assert np.allclose(holder_table([8.05502, -9.66459]), -19.2085)
    assert np.allclose(holder_table([-8.05502, -9.66459]), -19.2085)


def test_styblinski_tang():
    assert styblinski_tang([0]) == 0.0
    assert styblinski_tang([1]) == -5.0
        
    assert styblinski_tang([1, 2]) == -24.0
