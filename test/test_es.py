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
    def logging(self, logger, generation, solutions, function_values):
        assert 'best_f' in self.es.result
        if generation == 0:
            assert self.es.result['best_f'] == 9
        elif generation == 1:
            assert self.es.result['best_f'] == 16
        elif generation == 2:
            assert self.es.result['best_f'] == 25
        elif generation == 3:
            assert self.es.result['best_f'] == 36
        
        logger('generation', generation)
        logger('best_f', self.es.result['best_f'])
        
        return logger
        

class ESWorker(BaseESWorker):
    def __init__(self, master_conn, worker_conn):
        self.prepared = True
        super().__init__(master_conn, worker_conn)
        
    def f(self, config, solution):
        assert self.prepared
        assert config['train.num_iteration'] == 30
        
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
    es = ES()
    master = ESMaster(ESWorker, es, {'train.num_iteration': 30})
    logger = master(4)
    
    assert isinstance(logger.logs, dict)
    assert 'generation' in logger.logs
    assert logger.logs['generation'] == [0, 1, 2, 3]
    assert 'best_f' in logger.logs
    assert logger.logs['best_f'] == [9, 16, 25, 36]


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
