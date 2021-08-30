from itertools import product


class Grid(list):
    r"""Wrap a list of values to support a grid search. """
    def __init__(self, values):
        super().__init__(values)


class Sample(object):
    r"""Wrap a function to support a random search. """
    def __init__(self, f):
        assert callable(f)
        self.f = f

    def __call__(self):
        return self.f()


class Condition(object):
    r"""Conditional hyperparameter. """
    def __init__(self, f):
        assert callable(f)
        self.f = f
        
    def __call__(self, config):
        return self.f(config)


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property
    def logdir(self):
        return self._logdir
    
    @logdir.setter
    def logdir(self, logdir):
        self._logdir = logdir
        
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device):
        self._device = device


class Configurator(object):
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
        :func:`run_experiment`.
    
    Example::
    
        >>> config = Config({'env.id': Grid(['CartPole-v1', 'Ant-v2']), 'agent.lr': Grid([1e-3, 5e-3]), 'agent.size': [64, 64]})
        >>> configurator = Configurator(config, num_sample=1)
        >>> import pandas as pd
        >>> print(pd.DataFrame(configurator.make_configs()))
           ID       env.id  agent.lr agent.size
        0   0  CartPole-v1     0.001   [64, 64]
        1   1  CartPole-v1     0.005   [64, 64]
        2   2       Ant-v2     0.001   [64, 64]
        3   3       Ant-v2     0.005   [64, 64]

    Args:
        config (Config): a dictionary of all configuration items. 
        num_sample (int): number of samples for random configuration items. 
            If grid search is also provided, then the grid will be repeated :attr:`num_sample`
            of times. 

    .. _numerically unstable:
            http://cs231n.github.io/neural-networks-3/#hyper
    """
    def __init__(self, config, num_sample=1):
        if isinstance(config, dict) and not isinstance(config, Config):
            config = Config(config)
        assert isinstance(config, Config), f'expected type: Config, got {type(config)}'
        self.config = config
        self.num_sample = num_sample
        
    def make_configs(self):
        r"""Generate a list of all possible combinations of configurations, including
        grid search and random search. 
        
        Returns:
            list: a list of all possible configurations
        """
        keys_fixed = []
        keys_grid = []
        keys_sample = []
        for key, value in self.config.items():
            if isinstance(value, Grid):
                keys_grid.append(key)
            elif isinstance(value, Sample):
                keys_sample.append(key)
            else:
                keys_fixed.append(key)
        
        if len(keys_grid) == 0:
            all_grids = [{}]
        else:
            all_grids = product(*[self.config[key] for key in keys_grid])
            all_grids = [dict(zip(keys_grid, item)) for item in all_grids]
        if len(keys_sample) == 0:
            all_samples = [{}]
        else:
            all_samples = [{key: self.config[key]() for key in keys_sample} for _ in range(self.num_sample)]
        fixed_dict = {key: self.config[key] for key in keys_fixed}
        
        list_config = []
        for n, (grid_dict, sample_dict) in enumerate(product(all_grids, all_samples)):
            x = {**grid_dict, **sample_dict, **fixed_dict}
            x = {'ID': n, **{key: x[key] for key in self.config.keys()}}
            x = Config(x)
            for key, value in x.items():
                if isinstance(value, Condition):
                    x[key] = value(x)
            list_config.append(x)
        return list_config
