from pathlib import Path
from shutil import rmtree

import typing

import numpy as np

import pytest

from lagom.experiment import Configurator
from lagom.experiment import BaseExperimentWorker
from lagom.experiment import BaseExperimentMaster
from lagom.experiment import run_experiment

from lagom import BaseAlgorithm


class SimpleAlgorithm(BaseAlgorithm):
    def __call__(self, config, seed):
        return config['ID'], seed, f'ID: {config["ID"]}, seed: {seed}, Finished the work !'

    
class ExperimentWorker(BaseExperimentWorker):
    def make_algo(self):
        algo = SimpleAlgorithm(name='Simple algorithm')
        
        return algo
    
    
class ExperimentMaster(BaseExperimentMaster):
    def process_algo_result(self, config, seed, result):
        check_id, check_seed, msg = result
        assert isinstance(msg, str)
        assert check_id == config['ID']
        assert check_seed == seed
        
        print(msg)
        
    def make_configs(self):
        configurator = Configurator('grid')
        
        configurator.fixed('log.dir', 'some path')
        configurator.grid('network.lr', [0.1, 0.01, 0.05])
        configurator.grid('network.layers', [16, 32])
        configurator.grid('env.id', ['CartPole-v1', 'Ant-v2', 'HalfCheetah-v2'])
        
        configs = configurator.make_configs()

        return configs
    
    def make_seeds(self):
        return [123, 345, 567, 789, 901]
    
    
def test_configurator():
    # Construction invalidity check
    with pytest.raises(AssertionError):
        Configurator(search_mode='n')
    with pytest.raises(AssertionError):
        Configurator(search_mode='random', num_sample=None)

    # Create a configurator
    # Grid search
    configurator = Configurator(search_mode='grid')
    assert len(configurator.items) == 0

    with pytest.raises(AssertionError):
        configurator.fixed('seeds', [1, 2, 3])

    configurator.fixed('log.dir', 'some path')

    assert len(configurator.items) == 1
    assert isinstance(configurator.items['log.dir'], list)
    assert configurator.items['log.dir'][0] == 'some path'
    with pytest.raises(AssertionError):
        configurator.fixed('log.dir', 'second')
    with pytest.raises(AssertionError):
        configurator.grid('log.T', 'must be list, not string')
    configurator.grid('network.lr', [1e-2, 5e-3, 1e-4, 5e-4])
    configurator.grid('network.layers', [1, 2, 3])
    configurator.grid('env.id', ['CartPole-v1', 'Ant-v2'])

    configs = configurator.make_configs()

    assert len(configs) == 24
    # order-preserving check
    assert all([list(c.keys()) == ['ID', 'log.dir', 'network.lr', 'network.layers', 'env.id'] for c in configs])
    assert all([c['log.dir'] == 'some path' for c in configs])
    assert all([c['network.lr'] in [1e-2, 5e-3, 1e-4, 5e-4] for c in configs])
    assert all([c['network.layers'] in [1, 2, 3] for c in configs])
    assert all([c['env.id'] in ['CartPole-v1', 'Ant-v2'] for c in configs])

    # Grid search does not allow methods for random search
    with pytest.raises(AssertionError):
        configurator.categorical('one', [1, 2])
    with pytest.raises(AssertionError):
        configurator.uniform('two', 1, 3)
    with pytest.raises(AssertionError):
        configurator.discrete_uniform('three', 5, 10)
    with pytest.raises(AssertionError):
        configurator.log_uniform('four', 0.0001, 0.1)

    Configurator.print_config(configs[20])
    config_dataframe = Configurator.to_dataframe(configs)
    config_dataframe = Configurator.dataframe_subset(config_dataframe, 'network.lr', [0.01, 0.005])
    config_dataframe = Configurator.dataframe_groupview(config_dataframe, ['env.id', 'network.lr'])

    del configurator
    del configs
    del config_dataframe

    # Random search
    configurator = Configurator('random', num_sample=20)
    assert len(configurator.items) == 0

    with pytest.raises(AssertionError):
        configurator.fixed('seeds', [1, 2, 3])
    with pytest.raises(AssertionError):
        configurator.categorical('seeds', [1, 2])
    with pytest.raises(AssertionError):
        configurator.uniform('seeds', 1, 3)
    with pytest.raises(AssertionError):
        configurator.discrete_uniform('seeds', 5, 10)
    with pytest.raises(AssertionError):
        configurator.log_uniform('seeds', 0.0001, 0.1)

    with pytest.raises(AssertionError):
        configurator.grid('network.layers', [1, 2, 3])

    configurator.fixed('log.dir', 'some path')
    assert len(configurator.items) == 1
    assert isinstance(configurator.items['log.dir'], typing.Generator)
    assert next(configurator.items['log.dir']) == 'some path'
    with pytest.raises(AssertionError):
        configurator.fixed('log.dir', 'second')

    configurator.categorical('network.layers', [1, 2, 3])
    with pytest.raises(AssertionError):
        configurator.categorical('network.layers2', 12)  # must be list
    assert isinstance(configurator.items['network.layers'], typing.Generator)

    configurator.uniform('entropy_coef', 0.1, 2.0)
    configurator.discrete_uniform('train.N', 1, 100)
    configurator.log_uniform('network.lr', 1e-7, 1e-1)
    assert len(configurator.items) == 5

    configs = configurator.make_configs()

    assert len(configs[0]) == 1+5

    assert len(configs) == 20
    # order-preserving check
    l = ['ID', 'log.dir', 'network.layers', 'entropy_coef', 'train.N', 'network.lr']
    assert all([list(c.keys()) == l for c in configs])
    assert all([c['log.dir'] == 'some path' for c in configs])
    assert all([c['network.layers'] in [1, 2, 3] for c in configs])
    assert all([c['entropy_coef'] >= 0.1 and c['entropy_coef'] <= 2.0 for c in configs])
    assert all([isinstance(c['train.N'], int) for c in configs])
    assert all([c['train.N'] >= 1 and c['train.N'] <= 100 for c in configs])
    assert all([isinstance(c['network.lr'], float) for c in configs])
    assert all([c['network.lr'] >= 1e-7 and c['network.lr'] <= 1e-1 for c in configs])

    Configurator.print_config(configs[11])
    config_dataframe = Configurator.to_dataframe(configs)
    config_dataframe = Configurator.dataframe_subset(config_dataframe, 'network.layers', [1, 2])
    config_dataframe = Configurator.dataframe_groupview(config_dataframe, ['network.layers', 'log.dir'])
    config_dataframe
    
    
def test_experiment():
    experiment = ExperimentMaster(worker_class=ExperimentWorker, 
                                  max_num_worker=4, 
                                  daemonic_worker=None)

    assert len(experiment.configs) == 18
    assert len(experiment.seeds) == 5
    assert experiment.num_worker == 4
    assert experiment.num_iteration == 23
    
    experiment()
    
def test_run_experiment():
    run_experiment(worker_class=ExperimentWorker, 
                   master_class=ExperimentMaster, 
                   max_num_worker=None, 
                   daemonic_worker=None)
    
    # Sanity check of logging directories
    p = Path('./some path')
    assert p.exists()
    assert (p / 'configs.pkl').exists()
    # Check all configuration folders with their IDs and subfolders for all random seeds
    for i in range(18):
        config_p = p / str(i)
        assert config_p.exists()
        for seed in [123, 345, 567, 789, 901]:
            assert (config_p / str(seed)).exists

    # Clean the logging directory
    rmtree(p)
    
    # Test remove
    assert not p.exists()
