import numpy as np

import pytest

from lagom.experiment import Config


def test_config():
    config = Config()
    
    config.add_item(name='algo', val='RL')
    config.add_item(name='iter', val=30)
    config.add_item(name='hidden_sizes', val=[64, 32, 16])
    
    config.add_grid(name='cuda_num', val=[1, 2, 3])
    
    config.add_random_eps(name='lr', base=10, low=-6, high=0, num_sample=10)
    config.add_random_continuous(name='values', low=-5, high=5, num_sample=5)
    config.add_random_discrete(name='select', list_val=[43223, 5434, 21314], num_sample=10, replace=True)
    
    configs = config.make_configs()
    
    assert len(configs) == 1500
    
    assert np.alltrue([config['ID'] == i for i, config in enumerate(configs)])
    
    assert np.alltrue([config['algo'] == 'RL' for config in configs])
    
    assert np.alltrue(['iter' in config for config in configs])
    
    assert np.alltrue([config['hidden_sizes'] == [64, 32, 16] for config in configs])
    
    assert np.alltrue([config['cuda_num'] in [1, 2, 3] for config in configs])
    
    assert np.alltrue([config['lr'] > 0 and config['lr'] < 1 for config in configs])
    
    assert np.alltrue([config['values'] >= -5 and config['values'] < 5 for config in configs])
    
    assert np.alltrue([config['select'] in [43223, 5434, 21314] for config in configs])
    
    with pytest.raises(AssertionError):
        config.add_grid(name='wrong', val='yes')