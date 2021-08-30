from pathlib import Path
from shutil import rmtree

import numpy as np
import pytest

import torch
import torch.nn as nn
import torch.optim as optim

from lagom.experiment import Grid
from lagom.experiment import Sample
from lagom.experiment import Condition
from lagom.experiment import Config
from lagom.experiment import Configurator
from lagom.experiment import checkpointer
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


def test_config():
    def check(config):
        assert isinstance(config, Config)
        assert isinstance(config, dict)
        assert 'one' in config
        assert config['one'] == 1
        assert 'two' in config
        assert config['two'] == 2
        assert config.seed == 123 and config._seed == 123
        assert config.logdir == 'path' and config._logdir == 'path'
        assert config.device == 'cpu' and config._device == 'cpu'
    
    config = Config(one=1, two=2)
    config.seed = 123
    config.logdir = 'path'
    config.device = 'cpu'
    check(config)

    config = Config([['one', 1], ['two', 2]])
    config.seed = 123
    config.logdir = 'path'
    config.device = 'cpu'
    check(config)

    config = Config({'one': 1, 'two': 2})
    config.seed = 123
    config.logdir = 'path'
    config.device = 'cpu'
    check(config)


@pytest.mark.parametrize('num_sample', [1, 5, 10])
def test_configurator(num_sample):
    with pytest.raises(AssertionError):
        Configurator([1, 2, 3])
        
    config = Config({'log.dir': 'some path',
                     'beta': Condition(lambda x: 'small' if x['alpha'] < 5 else 'large'),
                     'network.type': 'MLP', 
                     'network.hidden_size': [64, 64],
                     'network.lr': Grid([1e-3, 1e-4]), 
                     'env.id': 'HalfCheetah-v2', 
                     'iter': Grid([10, 20, 30]),
                     'alpha': Sample(lambda: np.random.uniform(3, 10))})
    configurator = Configurator(config, num_sample=num_sample)
    list_config = configurator.make_configs()
    
    assert len(list_config) == 2*3*num_sample
    for ID in range(2*3*num_sample):
        assert list_config[ID]['ID'] == ID
        
    for x in list_config:
        assert isinstance(x, Config)
        assert len(x.keys()) == len(configurator.config.keys()) + 1  # added one more 'ID'
        for key in configurator.config.keys():
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
        
        assert list(x.keys()) == ['ID'] + list(configurator.config.keys())

    # test for non-random sampling
    config = Config({'log.dir': 'some path',
                     'network.type': 'MLP', 
                     'network.hidden_size': [64, 64],
                     'network.lr': Grid([1e-3, 1e-4]), 
                     'env.id': 'HalfCheetah-v2', 
                     'iter': Grid([10, 20, 30]),
                     'alpha': 0.1})
    configurator = Configurator(config, num_sample=num_sample)
    list_config = configurator.make_configs()
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
                     'alpha': 0.1})
    configurator = Configurator(config, num_sample=num_sample)
    list_config = configurator.make_configs()
    assert len(list_config) == 1  # no matter how many num_sample, without repetition
    for ID in range(1):
        assert list_config[ID]['ID'] == ID


def test_checkpointer():
    class Net(nn.Module):
        def __init__(self, lr):
            super().__init__()
            self.fc = nn.Linear(3, 4)
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
            self.register_buffer('total_step', torch.tensor(0))

        def forward(self, x):
            pass

    net1 = Net(lr=1e-3)
    net2 = Net(lr=5e-3)
    optimizer_out = optim.AdamW([*net1.parameters(), *net2.parameters()], lr=7e-4)
    loss = 12.5
    logs = [1, 2, 3, 4]

    config = Config({'nn.size': [64, 64], 'env.id': 'CartPole-v1'})
    config.seed = 0
    config.device = torch.device('cpu')
    config.logdir = Path('./')
    config.resume_checkpointer = config.logdir / 'resume_checkpoint.tar'

    checkpointer('save', config, obj=[loss, logs], state_obj=[net1, net2, optimizer_out])
    assert config.resume_checkpointer.exists()

    # Change some values before loading
    with torch.no_grad():
        net1.fc.weight.zero_()
        net2.fc.weight.zero_()
    loss = 0.1
    logs = []

    assert torch.allclose(net1.fc.weight, torch.tensor(0.0))
    assert torch.allclose(net2.fc.weight, torch.tensor(0.0))

    _loss, _logs = checkpointer('load', config, state_obj=[net1, net2, optimizer_out])

    # Check if loading works
    assert not torch.allclose(net1.fc.weight, torch.tensor(0.0))
    assert not torch.allclose(net2.fc.weight, torch.tensor(0.0))
    assert optimizer_out.state_dict()['param_groups'][0]['lr'] == 7e-4
    _loss = 12.5
    _logs = [1, 2, 3, 4]
    
    # remove the checkpoint file
    config.resume_checkpointer.unlink()


@pytest.mark.parametrize('num_sample', [1, 5])
@pytest.mark.parametrize('max_workers', [-1, None, 1, 5, 100])
@pytest.mark.parametrize('chunksize', [1, 7, 40])
def test_run_experiment(num_sample, max_workers, chunksize):
    def run(config):
        return config['ID'], config.seed, config.device, config.logdir, config.resume_checkpointer
    
    config = Config({'network.lr': Grid([1e-3, 5e-3]), 
                     'network.size': [32, 16],
                     'env.id': Grid(['CartPole-v1', 'Ant-v2'])})
    configurator = Configurator(config, num_sample=num_sample)
    seeds = [1, 2, 3]
    log_dir = './some_path'
    run_experiment(run, configurator, seeds, log_dir, max_workers, chunksize, use_gpu=False, gpu_ids=None)
 
    p = Path(log_dir)
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
