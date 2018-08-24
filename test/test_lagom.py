import pytest

import numpy as np

from lagom import Seeder


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
