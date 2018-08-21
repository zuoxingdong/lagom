import logging

from collections import OrderedDict

import numpy as np


class Logger(logging.Logger):
    """Logging information during experiment. 
    
    It supports iterative logging and dumping. That is, when same key is logged more than once, 
    the values for this key will be appended successively. During dumping, the user can also
    choose to dump either the entire list of logged values or the values with specific index.
    
    Note that we do not support hierarchical logging, e.g. list of dict of list of dict of ndarray
    this is because pickling is extremely slow for such a hierarhical data structure with mixture
    of dict and ndarray. Thus, we keep dict always at the top, if hierarchical logging is really
    needed, we recommand to present it in the key, the following example illustrate it:
    
    Suppose we want to train a goal-conditional policy in a maze with different goals iteratively,
    and each goal is trained with several internal iterations, in such scenario, when we want to 
    log policy loss, the hierarchical key can be combine into one string with ':' to separate each
    level, for example we want to log the policy loss with goal number 34 and internal training iteration
    20, the key can be 'goal_34:train:iter_20:policy_loss'. 
    """
    def __init__(self, name='logger'):
        """Initialize the Logger. 
        
        Args:
            name (str): name of the logger
        """
        super().__init__(name)
        
        self.name = name
        
        # Create logging dictionary, we use OrderedDict to keep insert ordering of the keys
        self.logs = OrderedDict()
        
    def log(self, key, val):
        """Log the information with given key and value. 
        
        Note that if key is already existed, the new value will be appended. 
        
        A recommandation for the string style of the key, it should be named semantically
        and each word separated by '_', because `dump()` will automatically replace all '_'
        with a whitespace and make each word capitalized by `str.title()`. 
        
        Args:
            key (str): key of the information
            val (object): value to be logged
        """
        # Initialize the logging with a list
        if key not in self.logs:
            self.logs[key] = []
            
        # Append the current value to be logged
        self.logs[key].append(val)
        
    def dump(self, keys=None, index=None, indent=0):
        """Dump the item to the screen.
        
        Args:
            keys (list, optional): List of keys to dump, if got None, then dump all logged information.
                Default: None
            index (int/list, optional): The index of logged information for dumping. It can be used with the 
                following types:
                1. Scalar: applies to all given keys. It can also be -1, i.e. last element in the list.
                2. List: applies to each key with given index. 
                3. None: dump everything for all given keys. It can also be list of None.
                Default: None
            indent (int, optional): the number of tab indentation before dumping the information. 
                Default: 0.
        """
        # Make keys depends on the cases
        if keys is None:  # dump all keys
            keys = list(self.logs.keys())
        assert isinstance(keys, list), f'keys must be list type, got {type(keys)}'
        
        # Make all indicies consistent with keys
        if index is None:  # dump everything in given keys
            index = ['all']*len(keys)
        if isinstance(index, int):  # apply to all given keys
            index = [index]*len(keys)
        elif isinstance(index, list):  # specific index for each key
            assert len(index) == len(keys), f'index length should be same as that of keys, got {len(index)}'
            index = index
        
        # Dump all logged information given the keys and index
        for key, idx in zip(keys, index):
            # Print given indentation
            if indent > 0:
                print('\t'*indent, end='')  # do not create a new line
            
            # Get logged information based on index
            if idx == 'all':
                log_data = self.logs[key]
            else:
                log_data = self.logs[key][idx]
            
            # Polish key string and make it visually beautiful
            key = key.strip().replace('_', ' ').title()
            
            # Print logged information
            print(f'{key}: {log_data}')

    def save(self, file=None):
        """Save loggings to a file
        
        Args:
            file (str): path to save the logged information. 
        """
        np.save(file, self.logs)
        
    def clear(self):
        """Remove all loggings"""
        self.logs.clear()
        
    def __repr__(self):
        return repr(self.logs)
