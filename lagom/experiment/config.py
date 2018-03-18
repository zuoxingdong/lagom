import numpy as np
from itertools import product

from collections import OrderedDict


class Config(object):
    """
    Base class of configurations for the experiments
    e.g. hyperparamters, environments/algorithm/logging settings
    """
    def __init__(self):
        self.config_settings = OrderedDict()
        self.configs = None
        
    def add(self, name, value):
        """
        Add an item for the configuration
        
        Args:
            name (str): the name for the configuration item
            value (list): the list of value(s) for the configuration item
        """
        if not isinstance(value, list):
            raise TypeError('The value must be of type list to support grid search. ')
        
        self.config_settings[name] = value
        
    def make_configs(self):
        """
        Generate a list of combinatino of configurations
        i.e. list of OrderedDict
        
        Returns:
            self.configs (list)
        """
        raise NotImplementedError
        
        
class GridConfig(Config):
    """
    Grid search based configurations
    """
    def make_configs(self):
        self.configs = []
        
        # Cartesian product (grid search) of all possible configurations
        for i, list_c in enumerate(product(*self.config_settings.values())):
            # Set up one configuration
            config = OrderedDict()
            config['ID'] = i
            for key, c in zip(self.config_settings.keys(), list_c):
                config[key] = c
            
            # Augment list of configurations
            self.configs.append(config)
            
        return self.configs
    
    
class RandomConfig(GridConfig):
    """
    Random search based configurations
    """
    def add_epsilon(self, name, base, low, high, num_sample):
        """
        Add a range of very small positive values to sample
        e.g. learning rates
        
        For numerical stability:
        http://cs231n.github.io/neural-networks-3/#hyper
        
        sample = base**np.random.uniform(low=low, high=high, size=num_sample)
        
        Args:
            num_sample (int): number of samples for this configuration
            name (str): name of configuration item
            base (float): base of exponential function
            low (float): lowest exponent
            high (float): highest exponent
        """
        super().add(name, (base**np.random.uniform(low=low, high=high, size=num_sample)).tolist())
    
    def add_continuous(self, name, low, high, num_sample):
        """
        Add a continuous range to sample
        
        Args:
            name (str): name of configuration item
            low (float): lowest value
            high (float): highest value
        """
        super().add(name, np.random.uniform(low=low, high=high, size=num_sample).tolist())
        
    def add_discrete(self, name, list_val, num_sample):
        """
        Add a discrete list of values to sample
        
        Args:
            name (str): name of configuration item
            list_val (list): list of discrete values to sample
        """
        super().add(name, np.random.choice(list_val, size=num_sample))
        
    def make_configs(self):
        return super().make_configs()