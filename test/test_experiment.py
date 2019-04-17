from pathlib import Path
from shutil import rmtree

import numpy as np

import pytest

from lagom.experiment import Grid
from lagom.experiment import Sample
from lagom.experiment import Condition
from lagom.experiment import Config
from lagom.experiment import ExperimentWorker
from lagom.experiment import ExperimentMaster
from lagom.experiment import run_experiment


@pytest.mark.parametrize('values', [[1, 2, 3], ['MLP', 'LSTM']])
def test_grid(values):
    grid = Grid(values)
    assert isinstance(grid, list)
    assert len(grid) == len(values)
    assert all([grid[i] == value for i, value in enumerate(values)])
    

def test_sample():
    sampler = Sample(lambda: 5)
    assert all([sampler() == 5 for _ in range(10)])
    del sampler
    
    sampler = Sample(lambda: np.random.uniform(3, 7))
    for _ in range(100):
        x = sampler()
        assert x >=3 and x< 7
        
        
def test_condition():
    condition = Condition(lambda x: 10 if x['one'] == 'ten' else -1)
    assert condition({'one': 'ten'}) == 10
    assert condition({'one': 10}) == -1


@pytest.mark.parametrize('num_sample', [1, 5, 10])
@pytest.mark.parametrize('keep_dict_order', [True, False])
def test_config(num_sample, keep_dict_order):
    with pytest.raises(AssertionError):
        Config([1, 2, 3])
        
    config = Config({'log.dir': 'some path',
                     'beta': Condition(lambda x: 'small' if x['alpha'] < 5 else 'large'),
                     'network.type': 'MLP', 
                     'network.hidden_size': [64, 64],
                     'network.lr': Grid([1e-3, 1e-4]), 
                     'env.id': 'HalfCheetah-v2', 
                     'iter': Grid([10, 20, 30]),
                     'alpha': Sample(lambda : np.random.uniform(3, 10))}, 
                    num_sample=num_sample, 
                    keep_dict_order=keep_dict_order)
    list_config = config.make_configs()
    
    assert len(list_config) == 2*3*num_sample
    for ID in range(2*3*num_sample):
        assert list_config[ID]['ID'] == ID
        
    for x in list_config:
        assert len(x.keys()) == len(config.items.keys()) + 1  # added one more 'ID'
        for key in config.items.keys():
            assert key in x
        assert x['log.dir'] == 'some path'
        if x['alpha'] < 5:
            assert x['beta'] == 'small'
        else:
            assert x['beta'] == 'large'
        assert x['network.type'] == 'MLP'
        assert x['network.hidden_size'] == [64, 64]
        assert x['network.lr'] in [1e-3, 1e-4]
        assert x['env.id'] == 'HalfCheetah-v2'
        assert x['iter'] in [10, 20, 30]
        assert x['alpha'] >= 3 and x['alpha'] < 10
        
        if keep_dict_order:
            assert list(x.keys()) == ['ID'] + list(config.items.keys())
        else:
            assert list(x.keys()) != ['ID'] + list(config.items.keys())
            assert list(x.keys()) == ['ID', 'log.dir', 'beta', 'network.type', 'network.hidden_size', 
                                      'env.id', 'network.lr', 'iter', 'alpha']
    
    # test for non-random sampling
    config = Config({'log.dir': 'some path',
                     'network.type': 'MLP', 
                     'network.hidden_size': [64, 64],
                     'network.lr': Grid([1e-3, 1e-4]), 
                     'env.id': 'HalfCheetah-v2', 
                     'iter': Grid([10, 20, 30]),
                     'alpha': 0.1}, 
                    num_sample=num_sample, 
                    keep_dict_order=keep_dict_order)
    list_config = config.make_configs()
    assert len(list_config) == 2*3*1  # no matter how many num_sample, without repetition
    for ID in range(2*3*1):
        assert list_config[ID]['ID'] == ID
        
    # test for all fixed
    config = Config({'log.dir': 'some path',
                     'network.type': 'MLP', 
                     'network.hidden_size': [64, 64],
                     'network.lr': 1e-3, 
                     'env.id': 'HalfCheetah-v2', 
                     'iter': 20,
                     'alpha': 0.1}, 
                    num_sample=num_sample, 
                    keep_dict_order=keep_dict_order)
    list_config = config.make_configs()
    assert len(list_config) == 1  # no matter how many num_sample, without repetition
    for ID in range(1):
        assert list_config[ID]['ID'] == ID


def run(config, seed, device):
        return config['ID'], seed


@pytest.mark.parametrize('num_sample', [1, 5])
@pytest.mark.parametrize('num_worker', [1, 3, 4, 7])
def test_experiment(num_sample, num_worker):
    config = Config({'log.dir': 'some path', 
                     'network.lr': Grid([1e-3, 5e-3]), 
                     'network.size': [32, 16],
                     'env.id': Grid(['CartPole-v1', 'Ant-v2'])}, 
                    num_sample=num_sample, 
                    keep_dict_order=True)
    experiment = ExperimentMaster(ExperimentWorker, num_worker=num_worker, run=run, config=config, seeds=[1, 2, 3])
    assert len(experiment.configs) == 4
    assert experiment.num_worker == num_worker
    
    tasks = experiment.make_tasks()
    assert len(tasks) == 4*3
    for task in tasks:
        assert isinstance(task[0], dict) and list(task[0].keys()) == ['ID'] + list(config.items.keys())
        assert task[1] in [1, 2, 3]
        assert task[2] is run
    
    results = experiment()
    assert len(results) == 4*3
    for i in range(0, 4*3, 3):
        assert results[i][0] == i//3
        assert results[i+1][0] == i//3
        assert results[i+2][0] == i//3
        
        assert results[i][1] == 1
        assert results[i+1][1] == 2
        assert results[i+2][1] == 3


@pytest.mark.parametrize('num_sample', [1, 5])
@pytest.mark.parametrize('num_worker', [1, 3, 4, 7])
def test_run_experiment(num_sample, num_worker):
    config = Config({'log.dir': 'some path', 
                     'network.lr': Grid([1e-3, 5e-3]), 
                     'network.size': [32, 16],
                     'env.id': Grid(['CartPole-v1', 'Ant-v2'])}, 
                    num_sample=num_sample, 
                    keep_dict_order=True)
    seeds = [1, 2, 3]
    run_experiment(run, config, seeds, num_worker)
    
    p = Path('./some path')
    assert p.exists()
    assert (p / 'configs.pkl').exists()
    # Check all configuration folders with their IDs and subfolders for all random seeds
    for i in range(4):
        config_p = p / str(i)
        assert config_p.exists()
        for seed in seeds:
            assert (config_p / str(seed)).exists
    # Clean the logging directory
    rmtree(p)
    # Test remove
    assert not p.exists()
