from time import time

from experiment import ExperimentWorker
from experiment import ExperimentMaster


t = time()

experiment = ExperimentMaster(worker_class=ExperimentWorker, 
                              num_worker=1, 
                              daemonic_worker=None)
experiment()

print(f'\nTotal time: {time() - t:.2} s')