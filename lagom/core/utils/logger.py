import logging
from pathlib import Path  # Unified path management, replace it with os package

from collections import OrderedDict

import numpy as np


class Logger(logging.Logger):
    """
    Logging information during experiment for all the configurations, each with a unique ID
    """
    def __init__(self, name='logger'):
        """
        Args:
            name (str): name of the logger
        """
        super().__init__(name)
        
        self.name = name
        
        # Create logging directory if it doesn't exist
        self.path = Path('logs')
        if not self.path.exists():
            self.path.mkdir()
            
        # Storage of loggings
        self.logs = OrderedDict()
        
    def log(self, config_ID, key_hierarchy, val):
        """
        Log information
        
        Args:
            config_ID (int): unique ID for specific configuration
            key_hierarchy (list): list of hierarchies of keys, e.g. int, str, tuple
            val (object): the value to be logged
        """
        # Make a alias of log, i.e. they share same memory
        log = self.logs
        # Iterate over hierarchies until the last second
        for key in [('ID', config_ID)] + key_hierarchy[:-1]:
            # Initialize if not existed
            if key not in log:
                log[key] = OrderedDict()
            # Capture hierarchy
            log = log[key]
            
        # Assign the last key with given value
        log[key_hierarchy[-1]] = val
        
    def dump(self, config_ID, key_hierarchy, indent=''):
        """
        Dump the item to the screen.
        
        Args:
            config_ID (int): unique ID for specific configuration
            key_hierarchy (list): list of hierarchies of keys; 
                                allowed: str or tuple of the form of (name, value)
            indent (str): the indentation before dumping any information
        """
        # Make a alias of log, i.e. they share same memory
        log = self.logs
        # Iterate over hierarchies
        for key in [('ID', config_ID)] + key_hierarchy:
            # Capture hierarchy
            log = log[key]
            
        # Print the item or all items under the dictionary
        # Note that only one level is supported
        key = key_hierarchy[-1]
        if isinstance(log, OrderedDict):
            print(f'{indent}{key}:')
            for k, v in log.items():
                print(f'{indent}\t{k}: {v}')
        else:
            print(f'{indent}{key}: {log}')

    def save(self):
        """Save loggings to a .npy file"""
        np.save(self.path/self.name, self.logs)
        
    def clear(self):
        """Remove all loggings"""
        self.logs.clear()