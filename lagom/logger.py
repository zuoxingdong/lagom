import logging

from collections import OrderedDict

from operator import itemgetter  # get list elements with arbitrary indices

import pickle


class Logger(logging.Logger):
    r"""Log the information of the experiment.
    
    It supports iterative logging and dumping. That is, when same key is logged more than once, 
    the values for this key will be appended successively. During dumping, the user can also
    choose to dump either the entire list of logged values or the values with specific index.
    
    .. note::
    
        It uses pickle to serialize the data. Empirically, ``pickle`` is 2x faster than ``numpy.save``
        and other alternatives like ``yaml`` is too slow and ``JSON`` does not support numpy array. 
    
    .. warning::
    
        It is highly discouraged to use hierarchical logging, e.g. list of dict of list of ndarray.
        This is because pickling such complex large data structure is extremely slow. It is recommended
        to use dictionary at topmost level
    
    Note that we do not support hierarchical logging, e.g. list of dict of list of dict of ndarray
    this is because pickling is extremely slow for such a hierarhical data structure with mixture
    of dict and ndarray. Thus, we keep dict always at the top, if hierarchical logging is really
    needed, we recommand to present it in the key, the following example illustrate it:
    
    Suppose we want to train a goal-conditional policy in a maze with different goals iteratively,
    and each goal is trained with several internal iterations, in such scenario, when we want to 
    log policy loss, the hierarchical key can be combine into one string with ':' to separate each
    level, for example we want to log the policy loss with goal number 34 and internal training iteration
    20, the key can be 'goal_34:train:iter_20:policy_loss'. 
    
    Example:
    
    * Default::
    
        >>> logger = Logger(name='logger')
        >>> logger.log('iteration', 1)
        >>> logger.log('training_loss', 0.12)
        >>> logger.log('iteration', 2)
        >>> logger.log('training_loss', 0.11)
        >>> logger.log('iteration', 3)
        >>> logger.log('training_loss', 0.09)
        >>> logger.dump()
        Iteration: [1, 2, 3]
        Training Loss: [0.12, 0.11, 0.09]
        
    * With indentation::
    
        >>> logger.dump(keys=None, index=None, indent=1)
        	Iteration: [1, 2, 3]
        	Training Loss: [0.12, 0.11, 0.09]
        
    * With specified keys::
    
        >>> logger.dump(keys=['iteration'], index=None, indent=0)
        Iteration: [1, 2, 3]
        
    * With specified single index::
    
        >>> logger.dump(keys=None, index=0, indent=0)
        Iteration: 1
        Training Loss: 0.12
        
    * With specified list of indices::
    
        >>> logger.dump(keys=None, index=[0, 2], indent=0)
        Iteration: [1, 3]
        Training Loss: [0.12, 0.09]
    
    """
    def __init__(self, name='logger'):
        r"""Initialize the Logger. 
        
        Args:
            name (str): name of the Logger
        """
        super().__init__(name)
        
        self.name = name
        
        # Create logging dictionary, we use OrderedDict to keep insert ordering of the keys
        self.logs = OrderedDict()
        
    def log(self, key, value):
        r"""Log the information with given key and value. 
        
        .. note::
        
            By default, each key is associated with a list. The list is created when using the key for
            the first time. All future loggings for this key will be appended to the list. 
            
        It is highly recommended to name the key string semantically and each word separated
        by '-', then :meth:`dump` will automatically replace all '-' with a whitespace and capitalize
        each word by ``str.title()``. 
        
        Args:
            key (str): key of the information
            value (object): value to be logged
        """
        if key not in self.logs:  # first time for this key, create a list
            self.logs[key] = []
            
        # Append the value
        self.logs[key].append(value)
        
    def dump(self, keys=None, index=None, indent=0):
        r"""Dump the loggings to the screen.
        
        Args:
            keys (list, optional): the list of selected keys to dump. If ``None``, then all keys will be used.
                Default: ``None``
            index (int/list, optional): the index in the list of each logged key to dump. If ``scalar``, then
                dumps all keys with given index and it can also be -1 to indicate the last element in the list.
                If ``list``, then dumps all keys with given indices. If ``None``, then dumps everything for all
                given keys. Default: ``None``
            indent (int, optional): the number of tab indentation before dumping the information. Default: 0
        """
        # Make keys depends on the cases
        if keys is None:  # dump all keys
            keys = list(self.logs.keys())
        assert isinstance(keys, list), f'expected list dtype, got {type(keys)}'
        
        # Make all indicies consistent with keys
        if index is None:  # dump everything in given keys
            index = ['all']*len(keys)
        elif isinstance(index, int):  # single index in given keys
            index = [index]*len(keys)
        elif isinstance(index, list):  # specific indices in given keys
            index = [index]*len(keys)
        
        # Dump all logged information given the keys and index
        for key, idx in zip(keys, index):
            # Print given indentation
            if indent > 0:
                print('\t'*indent, end='')  # do not create a new line
            
            # Get logged information based on index
            if idx == 'all':
                log_data = self.logs[key]
            elif isinstance(idx, int):  # single index
                log_data = self.logs[key][idx]
            elif isinstance(idx, list):  # specific indices
                log_data = list(itemgetter(*idx)(self.logs[key]))
            
            # Polish key string and make it visually beautiful
            key = key.strip().replace('_', ' ').title()
            
            # Print logged information
            print(f'{key}: {log_data}')

    def save(self, file):
        r"""Save loggings to a file using pickling. 
        
        Args:
            file (str): path to save the logged information. 
        """
        with open(file, 'wb') as f:
            pickle.dump(obj=self.logs, file=f, protocol=pickle.HIGHEST_PROTOCOL)
            
    @staticmethod
    def load(file):
        r"""Load loggings from a file using pickling. 
        
        Returns
        -------
        logging : OrderedDict
            Loaded logging dictionary
        """
        with open(file, 'rb') as f:
            logging = pickle.load(f)
            
        return logging
        
    def clear(self):
        r"""Remove all loggings in the dictionary. """
        self.logs.clear()
        
    def __repr__(self):
        return repr(self.logs)
