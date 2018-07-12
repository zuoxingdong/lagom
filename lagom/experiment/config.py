import numpy as np
from itertools import product

from collections import OrderedDict


class Config(object):
    """
    It defines the configurations for the experiments
    e.g. hyperparamters, environments/algorithm/logging settings
    
    All the configurations saved in a OrderedDict, and when there
    are too many configurations, it is then recommended to divide
    then into groups by adding tags in the key, e.g. {'network: lr': 1e-3}
    
    Note that it supports both grid search and random search. 
    """
    def __init__(self):
        self.config_settings = OrderedDict()
        self.configs = None
        
    def add_item(self, name, val):
        """
        Add an item for the configuration. 
        
        This can be used for grid search. 
        If val is a list, then auto-config generation will iterate each element in the list
        If val is not a list, then it is converted to a list as a single item. 
        
        Args:
            name (str): the name for the configuration item
            val (object/list): the value for the configuration item
        """
        assert name not in self.config_settings, 'The key is already existed. '
        
        # Convert non-list to list for auto-config generation (cartesian product)
        if not isinstance(val, list):
            val = [val]
        
        self.config_settings[name] = val
        
    def add_random_discrete(self, name, list_val, num_sample, replace=True):
        """
        Add a discrete list of values to sample from. 
        
        This can be used for random search sampled from given discrete list of items. 
        
        Args:
            name (str): name of configuration item
            list_val (list): list of discrete values to sample
            num_sample (int): number of samples
            replace (bool): If True, then sampling with replacement. Default: True
        """
        assert name not in self.config_settings, 'The key is already existed. '
        
        samples = np.random.choice(list_val, size=num_sample, replace=replace)
        samples = samples.tolist()
        
        self.config_settings[name] = samples
        
    def add_random_continuous(self, name, low, high, num_sample):
        """
        Add a continuous range to sample from uniformly. 
        
        This can be used for random search sampled from a given continuous range. 
        
        Args:
            name (str): name of configuration item
            low (float): lowest value
            high (float): highest value
            num_sample (int): number of samples
        """
        assert name not in self.config_settings, 'The key is already existed. '
        
        samples = np.random.uniform(low=low ,high=high, size=num_sample)
        samples = samples.tolist()
        
        self.config_settings[name] = samples
    
    def add_random_eps(self, name, base, low, high, num_sample):
        """
        Add a range of very small continuous positive values to sample from
        e.g. learning rates
        
        For numerical stability:
        http://cs231n.github.io/neural-networks-3/#hyper
        
        sample = base**np.random.uniform(low=low, high=high, size=num_sample)
        e.g. base=10, low=-6, high=0
        
        Args:
            name (str): name of configuration item
            base (float): base of exponential function
            low (float): lowest exponent
            high (float): highest exponent
            num_sample (int): number of samples
        """
        assert name not in self.config_settings, 'The key is already existed. '
        
        samples = base**np.random.uniform(low=low, high=high, size=num_sample)
        samples = samples.tolist()
        
        self.config_settings[name] = samples
    
    def make_configs(self):
        """
        Generate a list of combination of configurations
        i.e. list of OrderedDict, each augmented with a unique ID. 
        
        Returns:
            configs (list): list of configurations
        """
        configs = []
        
        # Cartesian product of all possible configurations
        for i, items in enumerate(product(*self.config_settings.values())):
            # Create one configuration
            config = OrderedDict()
            # Augment a unique ID
            config['ID'] = i
            # Add all configuration items
            for key, item in zip(self.config_settings.keys(), items):
                config[key] = item
                
            # Record configuration
            configs.append(config)
            
        # Record list of generated configurations
        self.configs = configs
            
        return configs
    
    def save_configs(self, f):
        """
        Save all generated configurations, an individual file for each configuration
        and named by its ID. 
        
        Args:
            f (str): path to save all configuration files
        """
        for config in self.configs:
            np.save(f'{f}/config_ID_{config["ID"]}', config)
           
    @staticmethod
    def print_config(config):
        """
        Helper function to print all items in the given configuration. 
        
        Args:
            config (dict): Dictionary of configurations. 
        """
        print(f'{"#":#^50}')
        [print(f'# {key}: {val}') for key, val in config.items()]
        print(f'{"#":#^50}')