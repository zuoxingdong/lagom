import numpy as np
from itertools import product

import pandas as pd

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
        Add a single item for the configuration. 
        
        It will be wrapped as a list with single element. 
        
        Args:
            name (str): the name for the configuration item
            val (object): the value for the configuration item
        """
        assert name not in self.config_settings, 'The key is already existed. '
        
        self.config_settings[name] = [val]
        
    def add_grid(self, name, val):
        """
        Add a list of configurations. 
        
        This can be used for grid search. 
        The auto-config generation will iterate over each element. 
        
        Args:
            name (str): the name for the configuration item
            val (list): a list of configuration items
        """
        assert name not in self.config_settings, 'The key is already existed. '
        
        assert isinstance(val, list)
        
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
        
        samples = np.random.uniform(low=low, high=high, size=num_sample)
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
        
    @staticmethod
    def to_pandas_dataframe(list_config):
        """
        Helper function to create a Pandas DataFrame for given list of configurations. 
        
        Args:
            list_config (list of dict): A list of configurations, each one is a dictionary. 
            
        Returns:
            config_dataframe (DataFrame): Pandas DataFrame for list of configurations
        """
        # Create a list of DataFrame, each for one configuration
        list_dataframe = [pd.DataFrame([config.values()], columns=config.keys()) for config in list_config]
        # Concatenate all DataFrame
        config_dataframe = pd.concat(list_dataframe, ignore_index=True)
        
        return config_dataframe
         
    @staticmethod
    def partition_IDs(config_dataframe, group_keys):
        """
        Reorganize DataFrame of all configurations by grouping some given keys in the combination 
        of all other keys and provide IDs for each group. 

        For example, if we want to plot training loss with different random seeds, although there are
        many other hyperparameter searchings, one can use the following to obtain IDs for different
        random seeds, under different other configurations:

            y, transformed_y = partition_IDs(x, group_keys=['seed'])

        In other words, the processed DataFrame will take a cartesian product of all other keys, 
        and each combination will contain IDs for all different random seeds. 

        We also return a transformed DataFrame that is more convenient for visualize the grouping
        in the Jupyter Notebook, it makes users easily verify how grouping is done. 

        To retrieve the corresponding IDs, we suggest the simplest way is to firstly visually check
        the transformed DataFrame and use the returned DataFrameGroupby object to get the data
        by calling `.groups.items()` and be aware that it returned the index for the original DataFrame, 
        not the ID number. However it could be used if all IDs are successive and starting from 0,
        i.e. consistent with DataFrame indicies. 

        Note that the key 'ID' should not be included in the group keys, because it will be added
        internally. 

        Args:
            config_dataframe (DataFrame): configuration DataFrame
            group_keys (list): list of keys to group up. 

        Returns:
            grouped_config (DataFrameGroupby): processed DataFrame
            transformed_config (DataFrame): useful for Notebook visualization
        """
        assert isinstance(config_dataframe, pd.DataFrame)
        assert 'ID' not in group_keys

        # Compute keys for Pandas method groupby
        # The keys should be complement of given group_keys
        # It has all keys except for 'ID' and group_keys
        keys = list(config_dataframe.keys())
        keys.remove('ID')
        [keys.remove(key) for key in group_keys]

        # Use Pandas method to group by keys
        grouped_config = config_dataframe.groupby(keys)

        # Make grouped keys be showing as a Series of dictionary
        transformed_config = grouped_config.apply(lambda x: pd.DataFrame({key: x[key].tolist() 
                                                                          for key in ['ID', *group_keys]}))

        return grouped_config, transformed_config
