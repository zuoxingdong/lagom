import logging

import os
import shutil

from collections import OrderedDict

import numpy as np


class Logger(logging.Logger):
    def __init__(self, name='logger', path='logs/', dump_mode=['screen']):
        super().__init__(name)
        
        self.dump_mode = dump_mode
        self.path = path
        # Create logging directory if it doesn't exist
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            
        # Loggings for some information, e.g. settings/hyperparameters
        self.infolog = OrderedDict()
        
        # Loggings for the metrics of training/evaluation
        self.metriclog = OrderedDict()  # keys: metric name
        
    def log_info(self, name, val):
        self.infolog[name] = val
        
    def log_metric(self, name, val, iter_num):
        # Initialize if such item not existed
        if name not in self.metriclog:
            self.metriclog[name] = OrderedDict()
            
        # Append a new value to the category
        self.metriclog[name][iter_num] = val
        
    def dump_metric(self, iter_num):
        if 'screen' in self.dump_mode:
            name_format = '\t{:<30}'  # align items in each iteration to the left
                
            print('{:->50}'.format(' '))  # separation line between each iteration
            print('Iteration # {:<d}'.format(iter_num))
            
            # Iterate over all metrics
            for name in self.metriclog.keys():
                [val, val_format] = self.metriclog[name][iter_num]
                print((name_format + val_format).format(name, val))
            
    def get_metriclog(self):
        return self.metriclog
    
    def get_path(self):
        return self.path
        
    def clear(self):
        self.infolog.clear()
        self.metriclog.clear()
            
    def remove_logfiles(self):
        """
        Remove everything in logging directory
        """
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path)