import numpy as np

import pytest

from lagom.experiment import Config
from lagom.experiment import BaseExperimentWorker
from lagom.experiment import BaseExperimentMaster

from lagom import BaseAlgorithm


class SimpleAlgorithm(BaseAlgorithm):
    def __call__(self, config):
        return config['ID'], 'Finish the work now !'
    
    
class ExperimentWorker(BaseExperimentWorker):
    def make_algo(self):
        algo = SimpleAlgorithm(name='Simple algorithm')
        
        return algo
    
    
class ExperimentMaster(BaseExperimentMaster):
    def process_algo_result(self, config, result):
        result, msg = result
        assert result == config['ID']
        
        print(msg)
        
    def make_configs(self):
        config = Config()

        config.add_item(name='iter', val=30)
        config.add_item(name='hidden_sizes', val=[64, 32, 16])
        config.add_random_eps(name='lr', base=10, low=-6, high=0, num_sample=10)
        config.add_random_continuous(name='values', low=-5, high=5, num_sample=5)
        config.add_random_discrete(name='select', list_val=[43223, 5434, 21314], num_sample=10, replace=True)

        configs = config.make_configs()

        return configs
    
    
def test_experiment():
    experiment = ExperimentMaster(worker_class=ExperimentWorker, 
                                  num_worker=128, 
                                  daemonic_worker=None)

    experiment()
    
    assert len(experiment.configs) <= experiment.num_iteration*experiment.num_worker
    assert len(experiment.configs) > (experiment.num_iteration - 1)*experiment.num_worker
    
    assert len(experiment.configs) == 500
    
    