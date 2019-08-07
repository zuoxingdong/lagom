from pathlib import Path
from shutil import rmtree

import numpy as np

import pytest

from lagom.experiment import Grid
from lagom.experiment import Sample
from lagom.experiment import Condition
from lagom.experiment import Config
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
        assert x >= 3 and x < 7
        
        
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
                     'alpha': Sample(lambda: np.random.uniform(3, 10))}, 
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


@pytest.mark.parametrize('num_sample', [1, 5])
@pytest.mark.parametrize('max_workers', [None, 1, 5, 100])
@pytest.mark.parametrize('chunksize', [1, 7, 40])
def test_run_experiment(num_sample, max_workers, chunksize):
    def run(config, seed, device, logdir):
        return config['ID'], seed, device, logdir
    
    config = Config({'network.lr': Grid([1e-3, 5e-3]), 
                     'network.size': [32, 16],
                     'env.id': Grid(['CartPole-v1', 'Ant-v2'])}, 
                    num_sample=num_sample, 
                    keep_dict_order=True)
    seeds = [1, 2, 3]
    log_dir = './some_path'
    run_experiment(run, config, seeds, log_dir, max_workers, chunksize, use_gpu=False, gpu_ids=None)
 
    p = Path('./some_path')
    assert p.exists()
    assert (p / 'configs.pkl').exists()
    assert (p / 'source_files').exists() and (p / 'source_files').is_dir()
    # Check all configuration folders with their IDs and subfolders for all random seeds
    for i in range(4):
        config_p = p / str(i)
        assert config_p.exists()
        assert (config_p / 'config.yml').exists()
        for seed in seeds:
            assert (config_p / str(seed)).exists()
    # Clean the logging directory
    rmtree(p)
    # Test remove
    assert not p.exists()
