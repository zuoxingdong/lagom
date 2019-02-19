import pytest

from lagom.multiprocessing import ProcessMaster
from lagom.multiprocessing import ProcessWorker


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
    
    
class Worker(ProcessWorker):
    def work(self, task_id, task):
        return naive_primality(task)
    
    
class Master(ProcessMaster):
    def make_tasks(self):
        primes = [16127, 23251, 29611, 37199]
        non_primes = [5853, 7179, 6957]
        tasks = [primes[0], primes[1], non_primes[0], primes[2], non_primes[1], non_primes[2], primes[3]]
        
        return tasks


@pytest.mark.parametrize('num_worker', [1, 2, 3, 5, 7, 10])
def test_process_master_worker(num_worker):
    def check(master):
        assert all([not p.is_alive() for p in master.list_process])
        assert all([conn.closed for conn in master.master_conns])
        assert all([conn.closed for conn in master.worker_conns])
    
    master = Master(Worker, num_worker)
    assert master.num_worker == num_worker
    results = master()
    assert results == [True, True, False, True, False, False, True]
    check(master)
