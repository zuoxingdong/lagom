from lagom.experiment import run_experiment

from experiment import ExperimentWorker
from experiment import ExperimentMaster


run_experiment(worker_class=ExperimentWorker, 
               master_class=ExperimentMaster, 
               max_num_worker=None, 
               daemonic_worker=None)
