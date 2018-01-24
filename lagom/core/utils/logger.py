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
        
    def log_metric(self, name, val, iter_num=None):
        # Initialize if such item not existed
        if name not in self.metriclog:
            self.metriclog[name] = OrderedDict()
            
        # Add a new value with given metric name, with iteration number if available
        if iter_num is not None:
            self.metriclog[name][iter_num] = val
        else:
            self.metriclog[name] = val
        
    def dump_metric(self, iter_num=None):
        #############
        # TODO: Now only support iteration as subcateory, make more generic dumping print
        #############
        if 'screen' in self.dump_mode:
            print('{:->60}'.format('-'))  # separation line
            
            if iter_num is not None:
                print('Iteration # {:<d}'.format(iter_num))
            
            # Print all metrics
            for name in self.metriclog.keys():
                if iter_num is not None:
                    val = self.metriclog[name][iter_num]
                    name_format = '\t{:<30}'
                else:
                    val = self.metriclog[name]
                    name_format = '{:<30}'
                    
                print((name_format + '{:<}').format(name, val))
    
    def save_metriclog(self, filename):
        """Save metric loggings to a .npy file"""
        np.save(os.path.join(self.path, filename), self.metriclog)
        
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