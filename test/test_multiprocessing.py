import numpy as np

import pytest

from lagom.multiprocessing import MPMaster
from lagom.multiprocessing import MPWorker


def naive_primality(integer):
    r"""Naive way to test a prime by iterating over all preceding integers. """
    prime = True
    if integer <= 1:
        prime = False
    else:
        for i in range(2, integer):
            if integer % i == 0:
                prime = False

    return prime
    
    
class Worker(MPWorker):
    def prepare(self):
        self.prepared = 'ok'
    
    def work(self, task):
        assert self.prepared == 'ok'
        
        task_id, task, use_chunk = task
        
        if use_chunk:
            result = [naive_primality(subtask) for subtask in task]
        else:
            result = naive_primality(task)
        
        return task_id, result
    
    
class Master(MPMaster):
    def make_tasks(self):
        primes = [14779693, 13956343, 20620837, 55449649]
        non_primes = [55449709, 20621087, 15608607]
        tasks = [primes[0], primes[1], non_primes[0], primes[2], non_primes[1], non_primes[2], primes[3]]
        
        return tasks
    
    def process_results(self, results):
        assert results == [True, True, False, True, False, False, True]
        print('pass')


def test_mp_master_worker():
    def check(master):
        assert all([not p.is_alive() for p in master.list_process])
        assert all([conn.closed for conn in master.master_conns])
        assert all([conn.closed for conn in master.worker_conns])
    
    master = Master(Worker, 9)
    assert master.num_worker == 9
    master()
    assert master.num_worker == 7
    check(master)
    del master
    
    master = Master(Worker, 5)
    assert master.num_worker == 5
    master()
    assert master.num_worker == 5
    check(master)
    del master
    
    master = Master(Worker, 3)
    assert master.num_worker == 3
    master()
    assert master.num_worker == 3
    check(master)
    del master
