import pytest

import torch
import numpy as np

import os
from pathlib import Path

from lagom import BaseAlgorithm
from lagom import Seeder
from lagom import Logger
from lagom import pickle_load
from lagom import pickle_dump
from lagom import yaml_load
from lagom import yaml_dump


class ToyAlgorithm(BaseAlgorithm):
    def __call__(self, config, seed, device):
        return config, seed, device


class TestLagom(object):
    def test_base_algorithm(self):
        algo = ToyAlgorithm()
        assert isinstance(algo, BaseAlgorithm)
        
        config, seed, device = algo(config={'lr': 0.1}, seed=1, device=torch.device('cpu'))
        
        assert isinstance(config, dict) and len(config) == 1
        assert 'lr' in config and config['lr'] == 0.1
        
        assert seed == 1
        assert device.type == 'cpu'
    
    def test_seeder(self):
        seeder = Seeder(init_seed=0)
        
        assert seeder.rng.get_state()[1][0] == 0
        assert np.random.get_state()[1][20] != seeder.rng.get_state()[1][20]
        
        # Single list of seeds
        seeds = seeder(size=1)
        assert len(seeds) == 1
        seeds = seeder(size=5)
        assert len(seeds) == 5
        
        # Batched seeds
        seeds = seeder(size=[1, 3])
        assert np.alltrue(np.array(seeds).shape == (1, 3))
        seeds = seeder(size=[2, 3])
        assert np.alltrue(np.array(seeds).shape == (2, 3))
        
    def test_pickle_yaml(self):
        a = {'one': 1, 'two': [2, 3]}
        b = {'three': 3, 'four': [4, 5]}
        c = [a, b]
        
        def _check(x):
            assert isinstance(x, list)
            assert len(x) == 2
            assert all([isinstance(i, dict) for i in x])
            assert list(x[0].keys()) == ['one', 'two']
            assert list(x[1].keys()) == ['three', 'four']
            assert list(x[0].values()) == [1, [2, 3]]
            assert list(x[1].values()) == [3, [4, 5]]
        
        # Pickle
        pickle_dump(c, '.tmp_pickle')
        _check(pickle_load('.tmp_pickle.pkl'))
        
        os.unlink('.tmp_pickle.pkl')
        
        # Yaml
        yaml_dump(c, '.tmp_yaml')
        _check(yaml_load('.tmp_yaml.yml'))
        
        os.unlink('.tmp_yaml.yml')
    
    def test_logger(self):
        logger = Logger()

        logger('iteration', 1)
        logger('learning_rate', 1e-3)
        logger('train_loss', 0.12)
        logger('eval_loss', 0.14)

        logger('iteration', 2)
        logger('learning_rate', 5e-4)
        logger('train_loss', 0.11)
        logger('eval_loss', 0.13)

        logger('iteration', 3)
        logger('learning_rate', 1e-4)
        logger('train_loss', 0.09)
        logger('eval_loss', 0.10)

        def check(logs):
            assert len(logs) == 4
            assert list(logs.keys()) == ['iteration', 'learning_rate', 'train_loss', 'eval_loss']
            assert logs['iteration'] == [1, 2, 3]
            assert np.allclose(logs['learning_rate'], [1e-3, 5e-4, 1e-4])
            assert np.allclose(logs['train_loss'], [0.12, 0.11, 0.09])
            assert np.allclose(logs['eval_loss'], [0.14, 0.13, 0.10])

        check(logger.logs)

        logger.dump()
        logger.dump(keys=['iteration'])
        logger.dump(keys=['iteration', 'train_loss'])
        logger.dump(index=0)
        logger.dump(index=[1, 2])
        logger.dump(index=0)
        logger.dump(keys=['iteration', 'eval_loss'], index=1)
        logger.dump(keys=['iteration', 'learning_rate'], indent=1)
        logger.dump(keys=['iteration', 'train_loss'], index=[0, 2], indent=1)

        f = Path('./logger_file')
        logger.save(f)
        f = f.with_suffix('.pkl')
        assert f.exists()

        logs = pickle_load(f)
        check(logs)

        f.unlink()
        assert not f.exists()

        logger.clear()
        assert len(logger.logs) == 0
