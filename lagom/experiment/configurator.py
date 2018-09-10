import numpy as np
from itertools import product

import pandas as pd


class Configurator(object):
    r"""Defines a set of configurations for the experiment. 
    
    The configuration includes the following possible items:
    
    * Hyperparameters: learning rate, batch size etc.
    
    * Experiment settings: training iterations, logging directory, environment name etc.
    
    .. note::
    
        All the items are stored in a ``dict`` (For Python 3.7+, it is order-preserving
        by default), each associated with a key and a list of possible values to iterate 
        over. It is a good practice to define the key with a categorized name separated
        by a dot, e.g. learning rate for neural network can use the key ``'network.lr'``. 
    
    Call :meth:`make_configs` to generate a list of all configurations, each is assigned
    with a unique ID. 
    
    It is useful for hyperparameter search, either grid search or random search. 
    
    .. note::
    
        The possible values for configuration items will be stored as a list for
        grid search and as a generator (infinite) for random search. 
        
    .. warning::
    
        The random seeds should not be set here. Instead, it should be handled by
        :class:`BaseExperimentMaster` and :class:`BaseExperimentWorker`.
    
    Example::
    
        >>> configurator = Configurator('grid')
        >>> configurator.fixed('log.dir', 'some path')
        >>> configurator.grid('network.lr', [1e-2, 5e-3])
        >>> configurator.grid('env.id', ['CartPole-v1', 'Ant-v2'])
        >>> list_config = configurator.make_configs()
        >>> Configurator.to_dataframe(list_config)
           ID    log.dir  network.lr       env.id
        0   0  some path       0.010  CartPole-v1
        1   1  some path       0.010       Ant-v2
        2   2  some path       0.005  CartPole-v1
        3   3  some path       0.005       Ant-v2
        
        >>> Configurator.dataframe_groupview(config_dataframe, ['env.id', 'network.lr'])
                                ID    log.dir
        env.id      network.lr               
        Ant-v2      0.005        3  some path
                    0.010        1  some path
        CartPole-v1 0.005        2  some path
                    0.010        0  some path
        
    """
    def __init__(self, search_mode, num_sample=None):
        r"""Initialize the configurator.
       
        Args:
            search_mode (str): the method to search for configuration, either 'grid' or 'random'.
            num_sample (int): the number of samples for :meth:`make_configs` to generate a list
                of possible configurations when the search mode is 'random'. 
        """
        self.items = {}
        
        modes = ['grid', 'random']
        assert search_mode in modes, f'expected {modes}, got {search_mode}'
        self.search_mode = search_mode
        
        if self.search_mode == 'random':
            assert num_sample is not None, 'in random search mode, num_sample cannot be None. '
        self.num_sample = num_sample
        
    def _redundancy_check(self, key):
        assert key not in self.items, f'the key {key} already existed in configuration items. '
        assert key != 'seeds', 'random seed should be handled by Experiment class !'
        
    def fixed(self, key, value):
        r"""A fixed configuration item without the need to search over. 
        
        .. note::
        
            For compatibility of :meth:`make_configs`, it wraps the value as a list of a single
            value for grid search and as a generator yielding this single value for random search.
        
        Args:
            key (str): the name of a configuration item. 
            value (object): the value for the configuration item.
        """
        self._redundancy_check(key)
        
        if self.search_mode == 'grid':
            self.items[key] = [value]
        elif self.search_mode == 'random':
            def make_generator():
                while True:
                    yield value
                    
            self.items[key] = make_generator()
        
    def grid(self, key, list_value):
        r"""An exhaustive grid search over a list of values for a specific configuration
        item. 
        
        .. note::
        
            For a single value without any search, one can simply put it in a list. 
        
        Args:
            key (str): the name of a configuration item. 
            list_value (list): a list of values for the configuration item. 
        """
        assert self.search_mode == 'grid', f'expected grid mode, got {self.search_mode}'
        self._redundancy_check(key)
        assert isinstance(list_value, list), f'expected list dtype, got {type(list_value)}'
        
        self.items[key] = list_value
        
    def categorical(self, key, list_value):
        r"""A random search sampled from a categorical distribution of a given list
        of values. 
        
        Args:
            key (str): the name of a configuration item. 
            list_value (list): a list of values for the configuration item.  
        """
        assert self.search_mode == 'random', f'expected random mode, got {self.search_mode}'
        self._redundancy_check(key)
        assert isinstance(list_value, list), f'expected list dtype, got {type(list_value)}'
        
        def make_generator():
            while True:
                yield list_value[np.random.choice(len(list_value))]
        
        self.items[key] = make_generator()
    
    def uniform(self, key, low, high):
        r"""A random search sampled uniformly from a continuous range. 
        
        Args:
            key (str): the name of a configuration item. 
            low (float/ndarray): lower bound
            high (float/ndarray): upper bound
        """
        assert self.search_mode == 'random', f'expected random mode, got {self.search_mode}'
        self._redundancy_check(key)
        
        def make_generator():
            while True:
                x = np.random.uniform(low=low, high=high)
                if np.isscalar(x):
                    yield float(x)
                else:
                    yield x.tolist()  # everything is float type, compatible to PyTorch
        
        self.items[key] = make_generator()
        
    def discrete_uniform(self, key, low, high):
        r"""A random search sampled uniformly from a discrete range over integers. 
        
        Args:
            key (str): the name of a configuration item. 
            low (float/ndarray): lowest possible integer (inclusive)
            high (float/ndarray): highest possible integer (exclusive)
        """
        assert self.search_mode == 'random', f'expected random mode, got {self.search_mode}'
        self._redundancy_check(key)
        
        def make_generator():
            while True:
                yield np.random.randint(low=low, high=high)
                
        self.items[key] = make_generator()
    
    def log_uniform(self, key, low, high):
        r"""A random search sampled from a log-uniform distribution given lower and upper
        bound. 
        
        The sampling process uses a log-uniform distribution, i.e.
        
        .. math::
            \text{logU}(a, b) \sim \exp(U(\log(a), \log(b)))
        
        .. note::
        
            This is useful to sample very small positive values (e.g. learning rate). 
            Note that direct uniform sampling is very `numerically unstable`_. 
        
        Args:
            key (str): the name of a configuration item. 
            low (float/ndarray): lower bound
            high (float/ndarray): upper bound
            
        .. _numerically unstable:
            http://cs231n.github.io/neural-networks-3/#hyper
        """
        assert self.search_mode == 'random', f'expected random mode, got {self.search_mode}'
        self._redundancy_check(key)
        
        def make_generator():
            while True:
                x = np.exp(np.random.uniform(low=np.log(low), high=np.log(high)))
                if np.isscalar(x):
                    yield float(x)
                else:
                    yield x.tolist()  # everything is float type, compatible to PyTorch
                
        self.items[key] = make_generator()
    
    def make_configs(self):
        r"""Generate a list of all possible combinations of configurations. 
        
        .. note::
        
            Each combination of configuration will be assigned a unique ID.
        
        Returns
        -------
        configs : list
            a list of all possible configurations
        """
        configs = []
        
        if self.search_mode == 'grid':
            # Cartesian product of all possible configuration values
            for i, items in enumerate(product(*self.items.values())):
                config = {}
                # Assign a unique ID
                config['ID'] = i
                # Add all configuration items
                for key, item in zip(self.items.keys(), items):
                    config[key] = item

                # Record configuration
                configs.append(config)
        elif self.search_mode == 'random':
            # Iterate over number of samples
            for i in range(self.num_sample):
                config = {}
                # Assign a unique ID
                config['ID'] = i
                # Add all configuration items
                for key, item_generator in self.items.items():
                    config[key] = next(item_generator)
                    
                # Record configuration
                configs.append(config)
        else:
            raise ValueError(f'expected either grid or random, got {self.search_mode}')
            
        return configs
               
    @staticmethod
    def print_config(config):
        r"""Print a configuration to the screen. 
        
        Args:
            config (dict): a dictionary of configuration items. 
        """
        print('#'*50)
        [print(f'# {key}: {value}') for key, value in config.items()]
        print('#'*50)
        
    @staticmethod
    def to_dataframe(list_config):
        r"""Create a Pandas DataFrame of the given list of configurations. 
        
        .. note::
        
            It is visually very convenient to display the DataFrame in Jupyter Notebook. 
        
        Args:
            list_config (list): A list of configurations, each one is a dictionary. 
            
        Returns
        -------
        config_dataframe : DataFrame
            a Pandas DataFrame for list of configurations
        """
        # Create a list of DataFrame, each for one configuration
        list_dataframe = [pd.DataFrame([config.values()], columns=config.keys()) for config in list_config]
        # Concatenate all DataFrame
        # Note that this is much more efficient than iteratively appending to DataFrame
        config_dataframe = pd.concat(list_dataframe, ignore_index=True)
        
        # Fill None object with string 'None', otherwise it will lead to problems
        # Note that it doesn't change configuration dictionary, algorithm runs correctly with None object
        config_dataframe.fillna(value='None', inplace=True)
        
        return config_dataframe
    
    @staticmethod
    def dataframe_subset(config_dataframe, key, list_value):
        r"""Take a subset of the given configuration DataFrame by selecting certain 
        values for a specific key. 
        
        Args:
            config_dataframe (DataFrame): a configuration DataFrame
            key (str): the key in the configuration for subset selection. 
            list_value (list): a list of selected values. 
            
        Returns
        -------
        subset : DataFrame
            selected subset of configuration DataFrame. 
        """
        assert isinstance(config_dataframe, pd.DataFrame), f'expected DataFrame, got {type(config_dataframe)}'
        assert isinstance(list_value, list)
        assert key in config_dataframe
        
        # Alias of subset
        subset = config_dataframe
        
        # Take subset
        subset = subset.loc[subset[key].isin(list_value)]
        
        return subset
    
    @staticmethod
    def dataframe_groupview(config_dataframe, list_key):
        r"""Reorganize the given configuration DataFrame by grouping items hierarchically
        with the given keys. 
        
        Args:
            config_dataframe (DataFrame): a configuration DataFrame
            list_key (list): a list of keys to group the DataFrame. 
            
        Returns
        -------
        grouped_dataframe : DataFrame
            a reorganized DataFrame with MultiIndex according to the given grouping keys. 
        """
        grouped_dataframe = config_dataframe.set_index(list_key)
        grouped_dataframe = grouped_dataframe.sort_index()
        
        return grouped_dataframe
