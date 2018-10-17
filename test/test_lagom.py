import pytest

import torch
import numpy as np

from pathlib import Path

from lagom import BaseAlgorithm
from lagom import Logger

from lagom.utils import pickle_load


class ToyAlgorithm(BaseAlgorithm):
    def __call__(self, config, seed, device):
        return config, seed, device


def test_base_algorithm():
    algo = ToyAlgorithm()
    assert isinstance(algo, BaseAlgorithm)
        
    config, seed, device = algo(config={'lr': 0.1}, seed=1, device=torch.device('cpu'))
        
    assert isinstance(config, dict) and len(config) == 1
    assert 'lr' in config and config['lr'] == 0.1
        
    assert seed == 1
    assert device.type == 'cpu'

    
def test_logger():
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
