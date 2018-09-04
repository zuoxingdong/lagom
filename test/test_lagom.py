import pytest

import numpy as np

import os

from lagom import Seeder
from lagom import Logger


class TestLagom(object):
    def test_utils(self):
        seeder = Seeder(init_seed=0)
        
        # Single list of seeds
        seeds = seeder(size=1)
        assert len(seeds) == 1
        seeds = seeder(size=5)
        assert len(seeds) == 5
        
        # Batch of seeds
        seeds = seeder(size=[1, 3])
        assert np.alltrue(np.array(seeds).shape == (1, 3))
        seeds = seeder(size=[2, 3])
        assert np.alltrue(np.array(seeds).shape == (2, 3))
    
    def test_logger(self):
        logger = Logger(name='logger')

        logger.log('iteration', 1)
        logger.log('learning_rate', 1e-3)
        logger.log('training_loss', 0.12)
        logger.log('evaluation_loss', 0.14)

        logger.log('iteration', 2)
        logger.log('learning_rate', 5e-4)
        logger.log('training_loss', 0.11)
        logger.log('evaluation_loss', 0.13)

        logger.log('iteration', 3)
        logger.log('learning_rate', 1e-4)
        logger.log('training_loss', 0.09)
        logger.log('evaluation_loss', 0.10)
        
        # Test dump, because dump will call print, impossible to use assert
        logger.dump()
        logger.dump(keys=None, index=None, indent=1)
        logger.dump(keys=None, index=None, indent=2)
        logger.dump(keys=['iteration', 'evaluation_loss'], index=None, indent=0)
        logger.dump(keys=None, index=0, indent=0)
        logger.dump(keys=None, index=2, indent=0)
        logger.dump(keys=None, index=[0, 2], indent=0)
        logger.dump(keys=['iteration', 'training_loss'], index=[0, 2], indent=0)
        
        # Test save function
        file = './test_logger_file'
        logger.save(file=file)
        
        assert os.path.exists(file)
        
        # Load file
        logging = Logger.load(file)
        
        assert len(logging) == 4
        assert 'iteration' in logging
        assert 'learning_rate' in logging
        assert 'training_loss' in logging
        assert 'evaluation_loss' in logging
        
        assert np.allclose(logging['iteration'], [1, 2, 3])
        assert np.allclose(logging['learning_rate'], [1e-3, 5e-4, 1e-4])
        assert np.allclose(logging['training_loss'], [0.12, 0.11, 0.09])
        assert np.allclose(logging['evaluation_loss'], [0.14, 0.13, 0.1])
        
        # Delete the temp logger file
        os.unlink(file)
