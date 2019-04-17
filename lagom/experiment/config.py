from itertools import product


class Grid(list):
    r"""A grid search over a list of values. """
    def __init__(self, values):
        super().__init__(values)


class Sample(object):
    def __init__(self, f):
        self.f = f
        
    def __call__(self):
        return self.f()
    
    
class Condition(object):
    def __init__(self, f):
        assert callable(f)
        self.f = f
        
    def __call__(self, config):
        return self.f(config)


class Config(object):
    r"""Defines a set of configurations for the experiment. 
    
    The configuration includes the following possible items:
    
    * Hyperparameters: learning rate, batch size etc.
    
    * Experiment settings: training iterations, logging directory, environment name etc.
    
    All items are stored in a dictionary. It is a good practice to semantically name each item
    e.g. `network.lr` indicates the learning rate of the neural network. 
    
    For hyperparameter search, we support both grid search (:class:`Grid`) and random search (:class:`Sample`).
    
    Call :meth:`make_configs` to generate a list of all configurations, each is assigned
    with a unique ID. 
    
    note::
    
        For random search over small positive float e.g. learning rate, it is recommended to
        use log-uniform distribution, i.e.
        .. math::
            \text{logU}(a, b) \sim \exp(U(\log(a), \log(b)))
        
        An example: `np.exp(np.random.uniform(low=np.log(low), high=np.log(high)))`
            
        Because direct uniform sampling is very `numerically unstable`_.
        
    .. warning::
    
        The random seeds should not be set here. Instead, it should be handled by
        :class:`BaseExperimentMaster` and :class:`BaseExperimentWorker`.
    
    Example::
    
        >>> config = Config({'log.dir': 'some path', 'network.lr': Grid([1e-3, 5e-3]), 'env.id': Grid(['CartPole-v1', 'Ant-v2'])}, num_sample=1, keep_dict_order=False)
        >>> import pandas as pd
        >>> print(pd.DataFrame(config.make_configs()))
               ID       env.id    log.dir  network.lr
            0   0  CartPole-v1  some path       0.001
            1   1       Ant-v2  some path       0.001
            2   2  CartPole-v1  some path       0.005
            3   3       Ant-v2  some path       0.005
    
    Args:
        items (dict): a dictionary of all configuration items. 
        num_sample (int): number of samples for random configuration items. 
            If grid search is also provided, then the grid will be repeated :attr:`num_sample`
            of times. 
        keep_dict_order (bool): if ``True``, then each generated configuration has the same
            key ordering with :attr:`items`. 
            
    .. _numerically unstable:
            http://cs231n.github.io/neural-networks-3/#hyper
    """
    def __init__(self, items, num_sample=1, keep_dict_order=False):
        assert isinstance(items, dict), f'dict type expected, got {type(items)}'
        self.items = items
        self.num_sample = num_sample
        self.keep_dict_order = keep_dict_order
        
    def make_configs(self):
        r"""Generate a list of all possible combinations of configurations, including
        grid search and random search. 
        
        Returns
        -------
        list_config : list
            a list of all possible configurations
        """
        keys_fixed = []
        keys_grid = []
        keys_sample = []
        for key in self.items.keys():
            x = self.items[key]
            if isinstance(x, Grid):
                keys_grid.append(key)
            elif isinstance(x, Sample):
                keys_sample.append(key)
            else:
                keys_fixed.append(key)
        if len(keys_sample) == 0:  # if no random search defined, set num_sample=1 to avoid repetition
            self.num_sample = 1
                
        product_grid = list(product(*[self.items[key] for key in keys_grid]))  # len >= 1, [()]
        list_config = []
        for n in range(len(product_grid)*self.num_sample):
            x = {'ID': n}
            x = {**x, **{key: self.items[key] for key in keys_fixed}}
            
            for idx, key in enumerate(keys_grid):
                x[key] = product_grid[n % len(product_grid)][idx]
            for key in keys_sample:
                x[key] = self.items[key]()
                
            if self.keep_dict_order:
                x = {**{'ID': x['ID']}, **{key: x[key] for key in self.items.keys()}}
                
            for key, value in x.items():
                if isinstance(value, Condition):
                    x[key] = value(x)
                
            list_config.append(x)
        return list_config
