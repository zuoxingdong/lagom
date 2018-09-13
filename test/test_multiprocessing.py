import numpy as np

import pytest

from lagom import Seeder

from lagom.core.multiprocessing import BaseWorker
from lagom.core.multiprocessing import BaseMaster
from lagom.core.multiprocessing import BaseIterativeMaster


def naive_primality(integer):
    """
    Naive way to test a prime by iterating
    over all preceding integers. 
    """
    prime = True
    if integer <= 1:
        prime = False
    else:
        for i in range(2, integer):
            if integer % i == 0:
                prime = False

    return prime
    
    
class NaivePrimalityWorker(BaseWorker):
    def work(self, master_cmd):
        task_id, task, seed = master_cmd
        
        result = []
        for integer in task:
            result.append(naive_primality(integer=integer))
        
        return task_id, result
    
    
class NaivePrimalityMaster(BaseMaster):
    def make_tasks(self):
        tasks = np.array_split(range(128*10), 128)
        
        return tasks
    
    def _process_workers_result(self, tasks, workers_result):
        for task, worker_result in zip(tasks, workers_result):
            task_id, result = worker_result
            for integer, prime in zip(task, result):
                assert prime == naive_primality(integer)


class NaivePrimalityIterativeMaster(BaseIterativeMaster):
    def make_tasks(self, iteration):
        tasks = np.array_split(range(128*10), self.num_worker)
        
        return tasks
    
    def _process_workers_result(self, tasks, workers_result):
        for task, worker_result in zip(tasks, workers_result):
            task_id, result = worker_result
            for integer, prime in zip(task, result):
                assert prime == naive_primality(integer)
    

class TestMultiprocessing(object):
    def test_seeder(self):
        seeder = Seeder(init_seed=0)
        
        assert seeder.rng.get_state()[1][0] == 0
        assert np.random.get_state()[1][20] != seeder.rng.get_state()[1][20]
        
        assert len(seeder(size=1)) == 1
        
        assert len(seeder(size=20)) == 20
        
        assert np.array(seeder(size=[2, 3, 4])).shape == (2, 3, 4)
        
    def test_master_worker(self):
        prime_test = NaivePrimalityMaster(worker_class=NaivePrimalityWorker, 
                                          num_worker=128, 
                                          init_seed=0, 
                                          daemonic_worker=None)
        prime_test()
        
    def test_iterative_master_worker(self):
        prime_test = NaivePrimalityIterativeMaster(num_iteration=3, 
                                                   worker_class=NaivePrimalityWorker, 
                                                   num_worker=128, 
                                                   init_seed=0, 
                                                   daemonic_worker=None)

        prime_test()
