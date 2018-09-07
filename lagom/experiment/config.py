import pickle

import numpy as np
from itertools import product

import pandas as pd

from collections import OrderedDict


class Config(object):
    r"""Defines the set of configurations for the experiment 
    e.g. hyperparamters, environments/algorithm/logging settings
    
    All the configurations are saved in a ``OrderedDict``, and when there are too 
    many configurations, it is then recommended to group the key by adding separation 
    symbol (semicolon, :), e.g. ``{'network:lr': 1e-3}``
    
    .. note::
    
        It supports both grid search and random search. 
    
    By calling :meth:`make_configs`, it will automatically generation a list of all
    possible configurations, each associated with a unique ID. In practice, it is also
    recommended to create an individual directory for each ID. 
    """
    def __init__(self):
        self.config_settings = OrderedDict()
        self.configs = None
        
    def add_item(self, name, val):
        r"""Add a single item in the configuration. 
        
        It will be wrapped as a list with single element. 
        
        Args:
            name (str): the name for the configuration item
            val (object): the value for the configuration item
            
        Example::
        
            >>> add_item(name='env:id', val='CartPole-v1')
            
        """
        assert name not in self.config_settings, 'The key is already existed. '
        
        self.config_settings[name] = [val]
        
    def add_grid(self, name, val):
        r"""Add a list of configurations. 
        
        .. note::
        
            This can be used for grid search. The :meth:`make_configs` will
            automatically iterate over each element in the list. 
        
        Args:
            name (str): the name for the configuration item
            val (list): a list of configuration items
        
        Example::
        
            >>> add_grid(name='lr', val=[1e-3, 5e-4, 1e-4])
            
        """
        assert name not in self.config_settings, 'The key is already existed. '
        
        assert isinstance(val, list)
        
        self.config_settings[name] = val
        
    def add_random_discrete(self, name, list_val, num_sample, replace=True):
        r"""Add a discrete list of values to sample from. 
        
        .. note::
        
            This can be used for random search. The items are sampled from
            the given list of possible values. 
        
        Args:
            name (str): name of configuration item
            list_val (list): list of possible values to sample from
            num_sample (int): number of samples
            replace (bool, optional): If True, then sampling with replacement. 
                Default: ``True``
                
        Example::
        
            >>> add_random_discrete(name='batch_size', list_val=[16, 32, 64], num_sample=2, replace=False)
            array([64, 32])
        
        """
        assert name not in self.config_settings, 'The key is already existed. '
        
        samples = np.random.choice(list_val, size=num_sample, replace=replace)
        samples = samples.tolist()
        
        self.config_settings[name] = samples
        
    def add_random_continuous(self, name, low, high, num_sample):
        r"""Add a continuous range to sample from. 
        
        .. note::
        
            This can be used for random search. The items are sampled from
            the given continuous range. 
        
        Args:
            name (str): name of configuration item
            low (float): lowest value
            high (float): highest value
            num_sample (int): number of samples
            
        Example::
        
            >>> add_random_continuous(name='entropy_coef', low=0, high=2, num_sample=2)
            array([0.55075687, 1.7957111 ])
        
        """
        assert name not in self.config_settings, 'The key is already existed. '
        
        samples = np.random.uniform(low=low, high=high, size=num_sample)
        samples = samples.tolist()
        
        self.config_settings[name] = samples
    
    def add_random_eps(self, name, base, low, high, num_sample):
        r"""Add a range of very small continuous positive values to sample from
        e.g. learning rates
        
        .. note::
        
            This can be useful to sample very small positive values like learning rate.
            Because direct uniform sampling leads to `numerical instability`_. 
        
        Args:
            name (str): name of configuration item
            base (float): base of exponential function
            low (float): lowest exponent
            high (float): highest exponent
            num_sample (int): number of samples
            
        Example::
        
            >>> add_random_eps(name='lr', base=10, low=-6, high=0, num_sample=2)
            array([1.56058072e-03, 2.18785186e-06])
            
        .. _numerical instability:
            http://cs231n.github.io/neural-networks-3/#hyper
        """
        assert name not in self.config_settings, 'The key is already existed. '
        
        samples = base**np.random.uniform(low=low, high=high, size=num_sample)
        samples = samples.tolist()
        
        self.config_settings[name] = samples
    
    def make_configs(self):
        r"""Generate a list of all possible combinations of configurations
        
        .. note::
        
            Each combination of configuration will be assigned a unique ID.
        
        Returns
        -------
        configs : list
            a list of all possible configurations
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
    
    def save_config_settings(self, f):
        r"""Save the settings of the configurations using pickle.
        
        .. note::
        
            Using pickle instead of ``numpy.save`` is because it is more convenient
            and faster for a small dictionary. 
            
        Args:
            f (str): file path to save the dictionary of configuration settings. 
        """
        pickle.dump()
    
    def save_configs(self, f):
        r"""Save all generated configurations, an individual file for each configuration
        and named by its ID. 
        
        Args:
            f (str): path to save all configuration files
        """
        for config in self.configs:
            np.save(f'{f}/config_ID_{config["ID"]}', config)
           
    @staticmethod
    def print_config(config):
        r"""Print all items for a given configuration. 
        
        Args:
            config (dict): a dictionary of configurations. 
        """
        print(f'{"#":#^50}')
        [print(f'# {key}: {val}') for key, val in config.items()]
        print(f'{"#":#^50}')
        
    @staticmethod
    def to_pandas_dataframe(list_config):
        r"""Create a Pandas DataFrame of the given list of configurations. 
        
        Args:
            list_config (list): A list of configurations, each one is a dictionary. 
            
        Returns
        -------
        config_dataframe : DataFrame
            Pandas DataFrame for list of configurations
        """
        # Create a list of DataFrame, each for one configuration
        list_dataframe = [pd.DataFrame([config.values()], columns=config.keys()) for config in list_config]
        # Concatenate all DataFrame
        # Note that this is much more efficient than iteratively appending to DataFrame
        config_dataframe = pd.concat(list_dataframe, ignore_index=True)
        
        # Fill None object with string 'None', otherwise it will lead to problems
        # Note that do not change it in original configuration dictionary, because it's important for algorithm
        # to run correctly with None object
        config_dataframe.fillna(value='None', inplace=True)
        
        return config_dataframe
    
    @staticmethod
    def subset_configs(config_dataframe, keyvalues):
        r"""Take a subset of given configurations based on the selection criterion for certain key and certain values. 
        
        For example, we want to select the configuration with seed only in the range of [20, 52, 70], and 
        learning rate in the range of [1e-3, 5e-4, 1e-4], one could do it as following:
        
            config_dataframe = config_dataframe.loc[config_dataframe['seed'].isin([20, 52, 70])]
            config_dataframe = config_dataframe.loc[config_dataframe['lr'].isin([1e-3, 5e-4, 1e-4])]
        
        Args:
            config_dataframe (DataFrame): configuration DataFrame
            keyvalues (dict): A dictionary of subset selection criterion. The keys corresponds to the columns in 
                the DataFrame. Each value should be a list of considered column values. 
        
        Returns
        -------
        subset : DataFrame
            selected subset of configurations as DataFrame.
        """
        assert isinstance(keyvalues, dict), f'expected dict dtype, but got {type(keyvalues)}'
        [isinstance(value, list) for value in keyvalues.values()]
        
        # Alias of subset
        subset = config_dataframe
        
        # Iterate over all given key values pairs
        for key, value_list in keyvalues.items():
            subset = subset.loc[subset[key].isin(value_list)]
            
        return subset
            
    @staticmethod
    def partition_IDs(config_dataframe, group_keys):
        r"""Reorganize DataFrame of all configurations by grouping some given keys in the combination 
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

        Returns
        -------
        grouped_config : DataFrameGroupby
            processed DataFrame
        transformed_config : DataFrame
            useful for Notebook visualization
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
        # This will be a displayable DataFrame with grouped information, very nice to visualize
        transformed_config = grouped_config.apply(lambda x: pd.DataFrame({key: x[key].tolist() 
                                                                          for key in ['ID', *group_keys]}))

        return grouped_config, transformed_config
